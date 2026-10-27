"""ik_trial_node.py — Registers `/ik_trial/start`, `/skip`, `/stop` and
periodically publishes `/ik_trial/status`.

Orchestrates `ik_action_node.IkActionManager` as a client — calling its
plain Python methods directly (see that module's docstring for why that's
equivalent to being a client of `/ik_action/*` here) and subscribing to
`/ik_action/feedback`/`/ik_action/result` the same way any external
observer would — rather than ever touching `/joint_states` or
`/joint_trajectory` itself, per the spec.

Samples random reachable targets by drawing from a bounding box sized to
the arm's total reach and rejecting draws `solve_ik` itself would reject —
"any distribution that reliably produces reachable targets is acceptable,"
per the spec.
"""

from __future__ import annotations

import logging
import random
import threading
from typing import Any, Dict, List, Optional, Tuple

from arm_runtime import ArmRuntime
from broker import Broker
from ik_action_node import IkActionManager
from kinematics import solve_ik
from services.provider import InProcessProvider, register as register_provider

log = logging.getLogger("ik_trial_node")

DEFAULT_DURATION = 30.0
STATUS_RATE_HZ = 10.0
_MAX_SAMPLE_ATTEMPTS = 200


class IkTrialNode:
    """Not a `stoppable` itself (see `_StatusThreadStopper` below) — its
    own `stop()` method means "end the running trial," the same name
    `/ik_trial/stop` uses, which would collide with main.py's generic
    shutdown-everything convention if this object were put in that list
    directly."""

    id = "ik_trial_node"

    def __init__(self, broker: Broker, runtime: ArmRuntime, manager: IkActionManager) -> None:
        self._broker = broker
        self._runtime = runtime
        self._manager = manager
        self._lock = threading.Lock()
        self._rng = random.Random()

        self._running = False
        self._duration = DEFAULT_DURATION
        self._epsilon: Optional[float] = None
        self._success_hold: Optional[float] = None
        self._start_sim_time = 0.0
        self._targets_reached = 0
        self._current_goal_id: Optional[str] = None
        self._target: Optional[Dict[str, float]] = None
        self._error: Optional[float] = None
        self._desired_positions: Optional[List[float]] = None
        self._action_status = "idle"

        broker.subscribe(self, "/ik_action/feedback")
        broker.subscribe(self, "/ik_action/result")

        self._stop_event = threading.Event()
        self._status_thread = threading.Thread(
            target=self._status_loop, name="ik-trial-status", daemon=True
        )
        self._status_thread.start()

    def stop_thread(self) -> None:
        self._stop_event.set()
        self._status_thread.join(timeout=2.0)

    # -- MessageSink interface (subscribed to /ik_action/feedback, /result)

    def send(self, message: dict) -> None:
        if message.get("op") != "publish":
            return
        topic = message.get("topic")
        msg = message.get("msg")
        if not isinstance(msg, dict):
            return
        if topic == "/ik_action/feedback":
            self._on_feedback(msg)
        elif topic == "/ik_action/result":
            self._on_result(msg)

    def _on_feedback(self, msg: dict) -> None:
        with self._lock:
            if msg.get("goal_id") != self._current_goal_id:
                return
            self._target = msg.get("target")
            self._error = msg.get("distance_remaining")
            self._desired_positions = msg.get("positions")
            self._action_status = "active"

    def _on_result(self, msg: dict) -> None:
        with self._lock:
            if not self._running or msg.get("goal_id") != self._current_goal_id:
                return
            outcome = msg.get("outcome")
            self._action_status = outcome
            if outcome == "reached":
                self._targets_reached += 1
        if outcome == "reached":
            self._submit_new_target()

    # -- services ----------------------------------------------------

    def start(
        self, duration: Optional[float], epsilon: Optional[float], success_hold: Optional[float]
    ) -> Tuple[bool, str]:
        self._manager.cancel_goal(None)  # discard any leftover goal; ignore result either way
        with self._lock:
            self._running = True
            self._duration = DEFAULT_DURATION if duration is None else duration
            self._epsilon = epsilon
            self._success_hold = success_hold
            self._targets_reached = 0
            self._start_sim_time = self._manager.get_sim_time()
            self._current_goal_id = None
            self._target = None
            self._error = None
            self._desired_positions = None
            self._action_status = "idle"

        if not self._submit_new_target():
            with self._lock:
                self._running = False
            return False, "no reachable target could be sampled"
        return True, ""

    def skip(self) -> Tuple[bool, str]:
        with self._lock:
            if not self._running:
                return False, "no trial is running"
        self._manager.cancel_goal(None)
        self._submit_new_target()
        return True, ""

    def stop(self) -> Tuple[bool, str]:
        with self._lock:
            if not self._running:
                return False, "no trial is running"
            self._running = False
        self._manager.cancel_goal(None)
        with self._lock:
            self._current_goal_id = None
            self._target = None
            self._error = None
            self._desired_positions = None
            self._action_status = "idle"
        return True, ""

    # -- internal ------------------------------------------------------

    def _sample_reachable_target(self) -> Optional[Tuple[float, float]]:
        _masses, lengths, _gravity = self._runtime.get_params()
        reach = sum(lengths)
        for _ in range(_MAX_SAMPLE_ATTEMPTS):
            x = self._rng.uniform(-reach, reach)
            y = self._rng.uniform(-reach, reach)
            try:
                solve_ik(lengths, x, y, None)
            except ValueError:
                continue
            return x, y
        return None

    def _submit_new_target(self) -> bool:
        target = self._sample_reachable_target()
        if target is None:
            return False
        x, y = target
        with self._lock:
            epsilon, success_hold = self._epsilon, self._success_hold
        goal_id, _error = self._manager.send_goal(x, y, None, epsilon, success_hold)
        if goal_id is None:
            return False
        with self._lock:
            self._current_goal_id = goal_id
            self._action_status = "active"
        return True

    def _status_loop(self) -> None:
        while not self._stop_event.wait(1.0 / STATUS_RATE_HZ):
            self._check_duration_elapsed()
            self._publish_status()

    def _check_duration_elapsed(self) -> None:
        with self._lock:
            if not self._running:
                return
            start_sim_time = self._start_sim_time
        # get_sim_time() is a call into the manager, not something to make
        # while holding this node's own lock (I/O-adjacent, per the
        # broker's own "lock for bookkeeping, never while calling out"
        # discipline).
        expired = (self._manager.get_sim_time() - start_sim_time) >= self._duration
        if expired:
            self.stop()

    def _publish_status(self) -> None:
        with self._lock:
            running = self._running
            start_sim_time = self._start_sim_time
        elapsed = (self._manager.get_sim_time() - start_sim_time) if running else 0.0
        with self._lock:
            msg = {
                "running": running,
                "elapsed": elapsed,
                "duration": self._duration,
                "targets_reached": self._targets_reached,
                "target": self._target,
                "error": self._error,
                "desired_positions": self._desired_positions,
                "action_status": self._action_status,
            }
        self._broker.publish("/ik_trial/status", msg)


class _StatusThreadStopper:
    """Adapts IkTrialNode's status-publishing thread to the generic
    `.stop()`/`.id` interface main.py's shutdown loop expects, without
    colliding with IkTrialNode's own domain-specific `stop()` (which ends
    a running trial, not the thread)."""

    def __init__(self, node: IkTrialNode) -> None:
        self.id = "ik_trial:status_thread"
        self._node = node

    def stop(self) -> None:
        self._node.stop_thread()


# --------------------------------------------------------------------------
# Services
# --------------------------------------------------------------------------


def _positive_number(args: dict, key: str) -> Tuple[Optional[float], Optional[str]]:
    value = args.get(key)
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None, f"args.{key} must be a positive number"
    return float(value), None


def _make_start_handler(node: IkTrialNode):
    def handler(args: Any) -> Tuple[Any, bool, str]:
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return {}, False, "args must be an object"

        duration, err = _positive_number(args, "duration")
        if err:
            return {}, False, err
        epsilon, err = _positive_number(args, "epsilon")
        if err:
            return {}, False, err
        success_hold_value = args.get("success_hold")
        if success_hold_value is not None and (
            isinstance(success_hold_value, bool)
            or not isinstance(success_hold_value, (int, float))
            or success_hold_value < 0
        ):
            return {}, False, "args.success_hold must be a nonnegative number"
        success_hold = None if success_hold_value is None else float(success_hold_value)

        ok, error = node.start(duration, epsilon, success_hold)
        return {}, ok, error

    return handler


def _make_skip_handler(node: IkTrialNode):
    def handler(args: Any) -> Tuple[Any, bool, str]:
        ok, error = node.skip()
        return {}, ok, error

    return handler


def _make_stop_handler(node: IkTrialNode):
    def handler(args: Any) -> Tuple[Any, bool, str]:
        ok, error = node.stop()
        return {}, ok, error

    return handler


def register(
    broker: Broker, runtime: ArmRuntime, manager: IkActionManager
) -> Tuple[List[Any], IkTrialNode]:
    node = IkTrialNode(broker, runtime, manager)
    stoppables: List[Any] = [
        register_provider(broker, "/ik_trial/start", _make_start_handler(node)),
        register_provider(broker, "/ik_trial/skip", _make_skip_handler(node)),
        register_provider(broker, "/ik_trial/stop", _make_stop_handler(node)),
        _StatusThreadStopper(node),
    ]
    return stoppables, node
