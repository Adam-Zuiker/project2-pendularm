"""Checks the /ik_action/* and /ik_trial/* transport contract — the scope
the spec itself limits grading to for these two nodes ("reachable/
unreachable goals accepted or rejected correctly, correct envelope
shapes... rather than waiting on real-time convergence").

Uses a real Broker + ArmRuntime (no sockets) so pub/sub and service
dispatch are exercised for real, just without the TCP layer.

Run with: python3 -m unittest tests.test_ik_action_and_trial -v
"""

from __future__ import annotations

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ik_action_node
import ik_trial_node
from arm_runtime import ArmRuntime
from broker import Broker


class _Capture:
    """A MessageSink that records every publish delivered to it."""

    id = "test-capture"

    def __init__(self) -> None:
        self.messages = []

    def send(self, message: dict) -> None:
        self.messages.append(message)


class TestIkAction(unittest.TestCase):
    def setUp(self):
        self.broker = Broker()
        self.runtime = ArmRuntime(2)
        self.runtime.set_params(masses=[1.0, 1.0], lengths=[1.0, 1.0])
        _providers, self.manager = ik_action_node.register(self.broker, self.runtime)
        self.capture = _Capture()
        self.broker.subscribe(self.capture, "/ik_action/result")
        self.broker.subscribe(self.capture, "/ik_action/feedback")

    def test_reachable_goal_accepted(self):
        goal_id, error = self.manager.send_goal(1.0, 0.5, None, None, None)
        self.assertIsNotNone(goal_id)
        self.assertEqual(error, "")

    def test_unreachable_goal_rejected_and_leaves_existing_goal_untouched(self):
        goal_id, _ = self.manager.send_goal(1.0, 0.5, None, None, None)
        bad_id, error = self.manager.send_goal(100.0, 0.0, None, None, None)
        self.assertIsNone(bad_id)
        self.assertNotEqual(error, "")
        # the original goal must still be cancelable -- i.e., still active
        ok, _ = self.manager.cancel_goal(goal_id)
        self.assertTrue(ok)

    def test_epsilon_must_be_positive(self):
        goal_id, error = self.manager.send_goal(1.0, 0.5, None, epsilon=-0.1, success_hold=None)
        self.assertIsNone(goal_id)
        self.assertIn("epsilon", error)

    def test_success_hold_must_be_nonnegative(self):
        goal_id, error = self.manager.send_goal(1.0, 0.5, None, epsilon=None, success_hold=-1.0)
        self.assertIsNone(goal_id)
        self.assertIn("success_hold", error)

    def test_cancel_with_no_active_goal_is_rejected(self):
        ok, error = self.manager.cancel_goal(None)
        self.assertFalse(ok)
        self.assertNotEqual(error, "")

    def test_cancel_with_wrong_goal_id_is_rejected(self):
        goal_id, _ = self.manager.send_goal(1.0, 0.5, None, None, None)
        ok, error = self.manager.cancel_goal("not-" + goal_id)
        self.assertFalse(ok)
        # the real goal must still be active
        ok2, _ = self.manager.cancel_goal(goal_id)
        self.assertTrue(ok2)

    def test_cancel_publishes_preempted_result(self):
        goal_id, _ = self.manager.send_goal(1.0, 0.5, None, None, None)
        self.manager.cancel_goal(goal_id)
        results = [m["msg"] for m in self.capture.messages if m.get("topic") == "/ik_action/result"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["goal_id"], goal_id)
        self.assertEqual(results[0]["outcome"], "preempted")

    def test_new_reachable_goal_preempts_the_old_one(self):
        goal1, _ = self.manager.send_goal(1.0, 0.5, None, None, None)
        goal2, _ = self.manager.send_goal(-1.0, 0.5, None, None, None)
        self.assertIsNotNone(goal2)
        results = [m["msg"] for m in self.capture.messages if m.get("topic") == "/ik_action/result"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["goal_id"], goal1)
        self.assertEqual(results[0]["outcome"], "preempted")
        # goal2 must be the one now active
        ok, _ = self.manager.cancel_goal(goal2)
        self.assertTrue(ok)

    def test_send_goal_publishes_joint_trajectory(self):
        traj_capture = _Capture()
        self.broker.subscribe(traj_capture, "/joint_trajectory")
        self.manager.send_goal(1.0, 0.5, None, None, None)
        published = [m["msg"] for m in traj_capture.messages if m.get("topic") == "/joint_trajectory"]
        self.assertEqual(len(published), 1)
        self.assertEqual(len(published[0]["points"][-1]["positions"]), 2)


class TestIkTrial(unittest.TestCase):
    def setUp(self):
        self.broker = Broker()
        self.runtime = ArmRuntime(2)
        self.runtime.set_params(masses=[1.0, 1.0], lengths=[1.0, 1.0])
        _ik_providers, self.manager = ik_action_node.register(self.broker, self.runtime)
        stoppables, self.node = ik_trial_node.register(self.broker, self.runtime, self.manager)
        self.stoppables = stoppables
        self.capture = _Capture()
        self.broker.subscribe(self.capture, "/ik_trial/status")

    def tearDown(self):
        for s in self.stoppables:
            s.stop()

    def _latest_status(self):
        statuses = [m["msg"] for m in self.capture.messages if m.get("topic") == "/ik_trial/status"]
        self.assertTrue(statuses, "no /ik_trial/status published yet")
        return statuses[-1]

    def _publish_joint_states_tick(self):
        """No physics loop is running in this test, so /joint_states is
        never published on its own -- IkActionManager only ticks (and
        therefore only emits /ik_action/feedback, which is what populates
        IkTrialNode's target/error/desired_positions) in response to one.
        Publish a single snapshot to simulate exactly one physics tick."""
        t, q, qdot, effort = self.runtime.get_joint_state()
        sec = int(t)
        nanosec = int(round((t - sec) * 1e9))
        self.broker.publish(
            "/joint_states",
            {
                "header": {"stamp": {"sec": sec, "nanosec": nanosec}, "frame_id": ""},
                "name": self.runtime.joint_names,
                "position": q,
                "velocity": qdot,
                "effort": effort,
            },
        )

    def test_skip_before_start_is_rejected(self):
        ok, error = self.node.skip()
        self.assertFalse(ok)
        self.assertNotEqual(error, "")

    def test_stop_before_start_is_rejected(self):
        ok, error = self.node.stop()
        self.assertFalse(ok)
        self.assertNotEqual(error, "")

    def test_start_submits_a_target_and_reports_active(self):
        ok, error = self.node.start(None, None, None)
        self.assertTrue(ok, error)
        self._publish_joint_states_tick()
        time.sleep(0.2)  # let the status thread publish at least once
        status = self._latest_status()
        self.assertTrue(status["running"])
        self.assertEqual(status["action_status"], "active")
        self.assertIsNotNone(status["target"])
        self.assertIsNotNone(status["desired_positions"])

    def test_skip_abandons_without_counting_and_submits_new_target(self):
        self.node.start(None, None, None)
        self._publish_joint_states_tick()
        time.sleep(0.2)
        first_target = self._latest_status()["target"]

        ok, _ = self.node.skip()
        self.assertTrue(ok)
        self._publish_joint_states_tick()
        time.sleep(0.2)
        status = self._latest_status()
        self.assertEqual(status["targets_reached"], 0)
        self.assertNotEqual(status["target"], first_target)

    def test_stop_clears_status_back_to_idle(self):
        self.node.start(None, None, None)
        time.sleep(0.2)
        ok, _ = self.node.stop()
        self.assertTrue(ok)
        time.sleep(0.2)
        status = self._latest_status()
        self.assertFalse(status["running"])
        self.assertEqual(status["action_status"], "idle")
        self.assertIsNone(status["target"])
        self.assertIsNone(status["error"])
        self.assertIsNone(status["desired_positions"])

    def test_stop_after_stop_is_rejected(self):
        self.node.start(None, None, None)
        self.node.stop()
        ok, error = self.node.stop()
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
