"""arm_lib/pid.py — Multi-joint PID controller, per
docs/PROJECT2_PENDULARM.md's "PID services" and "Control law" sections.

One `PidController` instance lives inside `ArmRuntime` (see arm_runtime.py),
which wraps every method here in its own lock — this class itself is not
thread-safe on its own and isn't meant to be used outside that wrapper.

`arm_runtime.py`'s physics tick only calls `compute_effort()` when
`is_enabled()` is true — while disabled, `tau` is zero and this class's
`compute_effort` is never invoked at all, which is how "PID starts disabled
by default" and "the integral doesn't advance while disabled" both hold
without `compute_effort` needing to check the enabled flag itself.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

Vector = List[float]

# Anti-windup clamp on the integral term. The spec: "Pick a reasonable
# fixed limit for your own implementation — its exact value doesn't
# matter, only that some bound exists." This is generous relative to the
# gravity-load torques this project's default masses/lengths produce.
INTEGRAL_LIMIT = 100.0


class PidController:
    def __init__(self, n: int) -> None:
        self._n = n
        self._kp: Vector = [0.0] * n
        self._ki: Vector = [0.0] * n
        self._kd: Vector = [0.0] * n
        self._integral: Vector = [0.0] * n
        self._enabled: bool = False
        self._setpoint_pos: Vector = [0.0] * n
        self._setpoint_vel: Vector = [0.0] * n

    def is_enabled(self) -> bool:
        return self._enabled

    def get_gains(self) -> Tuple[Vector, Vector, Vector]:
        return list(self._kp), list(self._ki), list(self._kd)

    def set_gains(
        self,
        kp: Optional[Vector] = None,
        ki: Optional[Vector] = None,
        kd: Optional[Vector] = None,
    ) -> Tuple[Vector, Vector, Vector, List[str]]:
        """Validate and apply each gain vector independently. Never touches
        the accumulated integral — per the spec, changing gains should show
        the effect of a nudge against the controller's current state, not a
        hidden reset every time you tune."""
        errors: List[str] = []
        for name, value, attr in (("kp", kp, "_kp"), ("ki", ki, "_ki"), ("kd", kd, "_kd")):
            if value is None:
                continue
            if len(value) != self._n:
                errors.append(f"{name} must have {self._n} entries")
            elif any(_negative_number(v) for v in value):
                errors.append(f"{name} must all be nonnegative numbers")
            else:
                setattr(self, attr, [float(v) for v in value])
        return list(self._kp), list(self._ki), list(self._kd), errors

    def set_setpoint(self, positions: Vector, velocities: Vector) -> None:
        self._setpoint_pos = list(positions)
        self._setpoint_vel = list(velocities)

    def get_setpoint(self) -> Tuple[Vector, Vector]:
        return list(self._setpoint_pos), list(self._setpoint_vel)

    # --------------------------------------------------------------------
    # TODO(human): implement enable(), reset(), and compute_effort() below
    # — the PID control law itself and its state-transition semantics.
    #
    # compute_effort(actual_pos, actual_vel, dt) is called once per physics
    # tick, only while enabled (arm_runtime.py's step() already checks
    # is_enabled() before calling this, so you can assume the controller is
    # enabled here). Apply this independently to every joint i:
    #
    #     position_error = setpoint_pos[i] - actual_pos[i]
    #     velocity_error  = setpoint_vel[i] - actual_vel[i]
    #     integral[i] = clamp(integral[i] + position_error*dt,
    #                          -INTEGRAL_LIMIT, +INTEGRAL_LIMIT)
    #     effort[i] = kp[i]*position_error + ki[i]*integral[i] + kd[i]*velocity_error
    #
    # The derivative term uses setpoint_vel - actual_vel (a velocity you
    # already have) rather than a finite difference of the position error
    # — see the spec's "Control law" section for why. This method should
    # both update self._integral in place and return the effort vector.
    #
    # enable(enabled) sets self._enabled. Re-enabling after a disable clears
    # the integral only — per the spec's "Control law" section, stated
    # twice: "clears out whatever integral accumulated while disabled" and
    # "should start from a clean integral... leave the integral alone [on
    # gain changes]". The setpoint is untouched by enable/disable; only
    # /arm_sim/reset (reset() below) clears it. Calling enable(True) while
    # already enabled shouldn't reset anything mid-run, only the
    # disabled->enabled transition itself does.
    #
    # reset() clears both the integral AND the setpoint back to zero — the
    # same all-zero reset pose arm_runtime.py's own reset() snaps (q, qdot)
    # to. Otherwise the freshly-reset arm would immediately get pulled back
    # toward whatever setpoint the controller still remembers.
    # --------------------------------------------------------------------

    def enable(self, enabled: bool) -> None:
        if enabled and not self._enabled:
            self._integral = [0.0] * self._n
        self._enabled = enabled

    def reset(self) -> None:
        self._setpoint_pos = [0.0] * self._n
        self._setpoint_vel = [0.0] * self._n
        self._integral = [0.0] * self._n


    def compute_effort(self, actual_pos: Vector, actual_vel: Vector, dt: float) -> Vector:
        effort = [0.0] * self._n
        for joint_idx in range(self._n):
            position_error = self._setpoint_pos[joint_idx] - actual_pos[joint_idx]
            velocity_error  = self._setpoint_vel[joint_idx] - actual_vel[joint_idx]
            self._integral[joint_idx] = max(min(self._integral[joint_idx] + position_error * dt, INTEGRAL_LIMIT), -INTEGRAL_LIMIT)
            effort[joint_idx] = self._kp[joint_idx] * position_error + self._ki[joint_idx] * self._integral[joint_idx] + self._kd[joint_idx] * velocity_error
        return effort

def _negative_number(value: object) -> bool:
    return isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
