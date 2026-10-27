"""Checks arm_lib.pid.PidController directly (control-law arithmetic,
integral clamping, enable/reset/set_gains state semantics) and, in
TestClosedLoopConvergence, the whole stack together — PID driving the real
n-link dynamics to a commanded setpoint, with steady-state effort checked
against the gravity load, per the spec's own suggested sanity check.

Run with: python3 -m unittest tests.test_pid -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arm_lib import dynamics
from arm_lib.pid import INTEGRAL_LIMIT, PidController
from arm_runtime import ArmRuntime


class TestControlLaw(unittest.TestCase):
    def test_formula_matches_hand_computed_effort(self):
        pid = PidController(2)
        pid.set_gains(kp=[10.0, 5.0], ki=[1.0, 0.5], kd=[2.0, 1.0])
        pid.enable(True)
        pid.set_setpoint([1.0, -0.5], [0.0, 0.0])

        effort = pid.compute_effort(actual_pos=[0.0, 0.0], actual_vel=[0.1, -0.2], dt=0.01)

        # position_error = [1.0, -0.5]; velocity_error = [-0.1, 0.2]
        # integral = [1.0*0.01, -0.5*0.01] = [0.01, -0.005] (starting from 0)
        expected = [
            10.0 * 1.0 + 1.0 * 0.01 + 2.0 * (-0.1),
            5.0 * (-0.5) + 0.5 * (-0.005) + 1.0 * 0.2,
        ]
        for actual, exp in zip(effort, expected):
            self.assertAlmostEqual(actual, exp, places=9)

    def test_integral_accumulates_across_calls(self):
        pid = PidController(1)
        pid.set_gains(kp=[0.0], ki=[1.0], kd=[0.0])
        pid.enable(True)
        pid.set_setpoint([1.0], [0.0])

        e1 = pid.compute_effort([0.0], [0.0], dt=0.1)
        e2 = pid.compute_effort([0.0], [0.0], dt=0.1)
        # Same position error each call (1.0), so the integral (and thus
        # effort, since kp=kd=0) should grow between calls, not reset.
        self.assertGreater(e2[0], e1[0])

    def test_integral_is_clamped(self):
        pid = PidController(1)
        pid.set_gains(kp=[0.0], ki=[1.0], kd=[0.0])
        pid.enable(True)
        pid.set_setpoint([1e6], [0.0])
        for _ in range(1000):
            effort = pid.compute_effort([0.0], [0.0], dt=1.0)
        self.assertLessEqual(effort[0], INTEGRAL_LIMIT * 1.0 + 1e-9)
        self.assertGreaterEqual(effort[0], -INTEGRAL_LIMIT * 1.0 - 1e-9)


class TestStateSemantics(unittest.TestCase):
    def test_enable_clears_integral_but_not_setpoint(self):
        # Per spec: re-enabling clears the accumulated integral only. The
        # setpoint is untouched by enable/disable -- only reset() clears it.
        pid = PidController(1)
        pid.set_gains(kp=[0.0], ki=[1.0], kd=[0.0])
        pid.set_setpoint([2.0], [0.0])
        pid.enable(True)
        pid.compute_effort([0.0], [0.0], dt=1.0)
        pid.compute_effort([0.0], [0.0], dt=1.0)  # integral now 4.0 (2.0*1.0 + 2.0*1.0)

        pid.enable(False)
        pid.enable(True)

        setpoint_pos, setpoint_vel = pid.get_setpoint()
        self.assertEqual(setpoint_pos, [2.0])
        self.assertEqual(setpoint_vel, [0.0])

        # Integral should be back at 0 -- a fresh call at the same error
        # (2.0) should show exactly one step's worth of accumulation
        # (ki=1.0, dt=1.0 -> integral becomes 2.0, effort = 1.0*2.0 = 2.0),
        # not the stale value from before the disable.
        effort_after = pid.compute_effort([0.0], [0.0], dt=1.0)
        self.assertAlmostEqual(effort_after[0], 2.0, places=6)

    def test_reset_clears_integral_and_setpoint(self):
        pid = PidController(2)
        pid.set_gains(kp=[0.0, 0.0], ki=[1.0, 1.0], kd=[0.0, 0.0])
        pid.set_setpoint([3.0, -3.0], [1.0, -1.0])
        pid.enable(True)
        pid.compute_effort([0.0, 0.0], [0.0, 0.0], dt=1.0)

        pid.reset()

        setpoint_pos, setpoint_vel = pid.get_setpoint()
        self.assertEqual(setpoint_pos, [0.0, 0.0])
        self.assertEqual(setpoint_vel, [0.0, 0.0])
        # With setpoint back at zero and actual also zero, error is zero,
        # so a fresh call should add nothing detectable to a cleared integral.
        effort = pid.compute_effort([0.0, 0.0], [0.0, 0.0], dt=1.0)
        self.assertAlmostEqual(effort[0], 0.0, places=9)
        self.assertAlmostEqual(effort[1], 0.0, places=9)

    def test_set_gains_does_not_reset_integral(self):
        pid = PidController(1)
        pid.set_gains(kp=[0.0], ki=[1.0], kd=[0.0])
        pid.set_setpoint([1.0], [0.0])
        pid.enable(True)
        pid.compute_effort([0.0], [0.0], dt=1.0)  # integral now 1.0

        pid.set_gains(kp=[5.0])  # tune kp; should not touch integral

        effort = pid.compute_effort([0.0], [0.0], dt=1.0)
        # kp*error (5*1=5) + ki*integral (1*(1.0+1.0)=2.0) = 7.0, only if
        # the integral carried over from before set_gains was called.
        self.assertAlmostEqual(effort[0], 7.0, places=6)


class TestClosedLoopConvergence(unittest.TestCase):
    def test_converges_to_setpoint_with_gravity_compensating_effort(self):
        runtime = ArmRuntime(2)
        runtime.set_params(masses=[1.0, 0.8], lengths=[1.0, 0.8], gravity=9.81)
        runtime.set_integrator(method="rk4", timestep=0.005)
        runtime.set_pid_gains(kp=[60.0, 40.0], ki=[15.0, 10.0], kd=[12.0, 8.0])
        setpoint = [0.6, -0.4]
        runtime.set_joint_trajectory_setpoint(setpoint, [0.0, 0.0])
        runtime.enable_pid(True)

        for _ in range(4000):  # 20s of simulated time at dt=0.005
            runtime.step()

        _, q, qdot, effort = runtime.get_joint_state()
        for actual, target in zip(q, setpoint):
            self.assertAlmostEqual(actual, target, delta=0.05)
        for v in qdot:
            self.assertAlmostEqual(v, 0.0, delta=0.1)

        masses, lengths, gravity = runtime.get_params()
        expected_gravity_load = dynamics.gravity_vector(masses, lengths, q, gravity)
        for actual_tau, expected_tau in zip(effort, expected_gravity_load):
            self.assertAlmostEqual(actual_tau, expected_tau, delta=0.5)


if __name__ == "__main__":
    unittest.main()
