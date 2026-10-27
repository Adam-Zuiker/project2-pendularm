"""PID controller checks (src/arm_lib/pid.py), run without the runtime."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from arm_lib.pid import INTEGRAL_LIMIT, PidController


def test_pid_control_law_gains_and_integral_clamp():
    pid = PidController(2)
    assert pid.is_enabled() is False, "PID must start disabled"
    _, _, _, errors = pid.set_gains(kp=[2.0, 1.0], ki=[0.5, 0.0], kd=[0.1, 3.0])
    assert errors == [], f"valid gains rejected: {errors}"
    pid.set_setpoint([1.0, -0.5], [0.2, 0.0])
    effort = pid.compute_effort([0.4, 0.5], [0.0, 1.0], 0.1)
    expected = [2.0 * 0.6 + 0.5 * 0.06 + 0.1 * 0.2, 1.0 * -1.0 + 0.0 + 3.0 * -1.0]
    assert all(abs(a - b) < 1e-12 for a, b in zip(effort, expected)), f"effort {effort}, expected {expected}"
    effort = pid.compute_effort([0.4, 0.5], [0.0, 1.0], 0.1)
    assert abs(effort[0] - (1.2 + 0.5 * 0.12 + 0.02)) < 1e-12, f"integral should accumulate to 0.12: {effort}"

    _, _, _, errors = pid.set_gains(kp=[1.0], ki=[-1.0, 0.0], kd=[0.0, True])
    assert len(errors) == 3, f"wrong-length, negative and boolean gains must each be rejected: {errors}"
    assert pid.get_gains() == ([2.0, 1.0], [0.5, 0.0], [0.1, 3.0]), f"rejected gains changed state: {pid.get_gains()}"

    clamp = PidController(1)
    clamp.set_gains(kp=[0.0], ki=[1.0], kd=[0.0])
    clamp.set_setpoint([1e6], [0.0])
    clamp.compute_effort([0.0], [0.0], 1.0)
    assert clamp.compute_effort([0.0], [0.0], 1.0) == [INTEGRAL_LIMIT], "integral must clamp at +INTEGRAL_LIMIT"
    clamp.set_setpoint([-1e6], [0.0])
    for _ in range(3):
        effort = clamp.compute_effort([0.0], [0.0], 1.0)
    assert effort == [-INTEGRAL_LIMIT], f"integral must clamp at -INTEGRAL_LIMIT, got {effort}"


def test_pid_enable_and_reset_semantics():
    pid = PidController(2)
    pid.set_gains(kp=[0.0, 0.0], ki=[1.0, 1.0], kd=[0.0, 0.0])
    pid.set_setpoint([1.0, 2.0], [0.0, 0.0])
    pid.enable(True)
    pid.compute_effort([0.0, 0.0], [0.0, 0.0], 0.5)
    pid.enable(True)
    effort = pid.compute_effort([0.0, 0.0], [0.0, 0.0], 0.5)
    assert effort == [1.0, 2.0], f"enable(True) while already enabled must keep the integral: effort {effort}, expected [1.0, 2.0]"

    pid.enable(False)
    assert pid.is_enabled() is False, "enable(False) must disable the controller"
    assert pid.get_setpoint() == ([1.0, 2.0], [0.0, 0.0]), f"disable must not touch the setpoint: {pid.get_setpoint()}"
    pid.enable(True)
    assert pid.is_enabled() is True, "enable(True) must enable the controller"
    effort = pid.compute_effort([0.0, 0.0], [0.0, 0.0], 0.5)
    assert effort == [0.5, 1.0], f"re-enabling after a disable must clear the integral: effort {effort}, expected [0.5, 1.0]"

    pid.reset()
    assert pid.get_setpoint() == ([0.0, 0.0], [0.0, 0.0]), f"reset must zero the setpoint: {pid.get_setpoint()}"
    effort = pid.compute_effort([0.0, 0.0], [0.0, 0.0], 0.5)
    assert effort == [0.0, 0.0], f"reset must clear the integral and setpoint: effort {effort}, expected [0.0, 0.0]"
    assert pid.get_gains() == ([0.0, 0.0], [1.0, 1.0], [0.0, 0.0]), f"reset must not change gains: {pid.get_gains()}"
