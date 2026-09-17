"""arm_runtime.py — ArmRuntime: owns the live n-link arm's (q, qdot), link
params, integrator/timestep choice, sim clock, and pause flag, all behind
one lock — the same "lock only for bookkeeping, never for I/O" discipline
broker.py's Broker uses (see that module's docstring for the full argument;
it applies unchanged here).

`step()` applies zero effort whenever the PID controller is disabled —
which is exactly the spec's "PID starts disabled by default" behavior, so
the arm swings freely under gravity until a client explicitly enables
control. The controller itself (`arm_lib.pid.PidController`) is owned here
and protected by the same lock as everything else in this class.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

from arm_lib import dynamics, linalg
from arm_lib.integrators import INTEGRATORS
from arm_lib.pid import PidController

Vector = List[float]

DEFAULT_GRAVITY = 9.81
# Neither the spec nor the layout doc mandates a default integrator/timestep
# for the live arm (only /arm_sim/integration_step's request is required to
# name one explicitly) — rk4 at a moderate timestep is a reasonable,
# stable-by-default starting point until a client calls set_integrator.
DEFAULT_INTEGRATOR = "rk4"
DEFAULT_TIMESTEP = 0.01


class ArmRuntime:
    def __init__(self, n_links: int) -> None:
        self._lock = threading.Lock()
        self._n = n_links
        self._masses: Vector = [1.0] * n_links
        self._lengths: Vector = [1.0] * n_links
        self._gravity: float = DEFAULT_GRAVITY
        self._integrator_name: str = DEFAULT_INTEGRATOR
        self._dt: float = DEFAULT_TIMESTEP
        self._paused: bool = False
        self._t: float = 0.0
        # Reset pose: all joints at 0 (arm fully extended along +x). With
        # nonzero gravity this is not an equilibrium, so the spec's "the arm
        # should visibly move as soon as the runtime starts" holds.
        self._q: Vector = [0.0] * n_links
        self._qdot: Vector = [0.0] * n_links
        self._last_effort: Vector = [0.0] * n_links
        self._pid = PidController(n_links)

    @property
    def n_links(self) -> int:
        return self._n

    @property
    def joint_names(self) -> List[str]:
        return [f"joint{i + 1}" for i in range(self._n)]

    @property
    def dt(self) -> float:
        with self._lock:
            return self._dt

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def set_pause(self, paused: Optional[bool]) -> bool:
        """`paused=None` queries without changing anything (the spec's
        empty-request convention). Returns the resulting, possibly
        unchanged, paused state."""
        with self._lock:
            if paused is not None:
                self._paused = paused
            return self._paused

    def get_joint_state(self) -> Tuple[float, Vector, Vector, Vector]:
        with self._lock:
            return self._t, list(self._q), list(self._qdot), list(self._last_effort)

    def get_params(self) -> Tuple[Vector, Vector, float]:
        with self._lock:
            return list(self._masses), list(self._lengths), self._gravity

    def set_params(
        self,
        masses: Optional[Vector] = None,
        lengths: Optional[Vector] = None,
        gravity: Optional[float] = None,
    ) -> Tuple[Vector, Vector, float, List[str]]:
        """Validate and apply each field independently. Always returns the
        current post-update `(masses, lengths, gravity)`, plus a list of
        human-readable rejection reasons for whichever fields were invalid
        (empty if every field that was provided got applied)."""
        errors: List[str] = []
        with self._lock:
            if masses is not None:
                if len(masses) != self._n:
                    errors.append(f"masses must have {self._n} entries")
                elif any(_not_positive_number(m) for m in masses):
                    errors.append("masses must all be positive numbers")
                else:
                    self._masses = [float(m) for m in masses]

            if lengths is not None:
                if len(lengths) != self._n:
                    errors.append(f"lengths must have {self._n} entries")
                elif any(_not_positive_number(l) for l in lengths):
                    errors.append("lengths must all be positive numbers")
                else:
                    self._lengths = [float(l) for l in lengths]

            if gravity is not None:
                if isinstance(gravity, bool) or not isinstance(gravity, (int, float)) or gravity < 0:
                    errors.append("gravity must be a nonnegative number")
                else:
                    self._gravity = float(gravity)

            return list(self._masses), list(self._lengths), self._gravity, errors

    def set_integrator(
        self, method: Optional[str] = None, timestep: Optional[float] = None
    ) -> Tuple[str, float, List[str]]:
        errors: List[str] = []
        with self._lock:
            if method is not None:
                if method not in INTEGRATORS:
                    errors.append(f"unknown integrator {method!r}")
                else:
                    self._integrator_name = method

            if timestep is not None:
                if isinstance(timestep, bool) or not isinstance(timestep, (int, float)) or timestep <= 0:
                    errors.append("timestep must be positive")
                else:
                    self._dt = float(timestep)

            return self._integrator_name, self._dt, errors

    def reset(self) -> Tuple[Vector, Vector]:
        with self._lock:
            self._q = [0.0] * self._n
            self._qdot = [0.0] * self._n
            self._t = 0.0
            self._pid.reset()
            return list(self._q), list(self._qdot)

    # -- PID -------------------------------------------------------------

    def is_pid_enabled(self) -> bool:
        with self._lock:
            return self._pid.is_enabled()

    def enable_pid(self, enabled: Optional[bool]) -> bool:
        """`enabled=None` queries without changing anything (the spec's
        empty-request convention, same as set_pause)."""
        with self._lock:
            if enabled is not None:
                self._pid.enable(enabled)
            return self._pid.is_enabled()

    def get_pid_gains(self) -> Tuple[Vector, Vector, Vector]:
        with self._lock:
            return self._pid.get_gains()

    def set_pid_gains(
        self,
        kp: Optional[Vector] = None,
        ki: Optional[Vector] = None,
        kd: Optional[Vector] = None,
    ) -> Tuple[Vector, Vector, Vector, List[str]]:
        with self._lock:
            return self._pid.set_gains(kp, ki, kd)

    def set_joint_trajectory_setpoint(self, positions: Vector, velocities: Vector) -> None:
        with self._lock:
            self._pid.set_setpoint(positions, velocities)

    def step(self) -> None:
        """Advance the simulation by exactly one configured timestep.
        Called once per physics-thread tick (see arm_sim_node.py) when not
        paused — pause semantics live entirely in that caller's loop, not
        here, so this method has no pause check of its own."""
        with self._lock:
            masses, lengths, gravity = self._masses, self._lengths, self._gravity
            integrator_name, dt = self._integrator_name, self._dt
            t, q, qdot = self._t, self._q, self._qdot

            if self._pid.is_enabled():
                tau = self._pid.compute_effort(q, qdot, dt)
            else:
                tau = [0.0] * self._n

            def qddot_func(t_eval: float, q_eval: Vector, qdot_eval: Vector) -> Vector:
                mass_matrix = dynamics.mass_matrix(masses, lengths, q_eval)
                coriolis = dynamics.coriolis_forces(masses, lengths, q_eval, qdot_eval)
                gravity_load = dynamics.gravity_vector(masses, lengths, q_eval, gravity)
                rhs = linalg.vec_sub(linalg.vec_sub(tau, coriolis), gravity_load)
                return linalg.solve_linear_system(mass_matrix, rhs)

            step_fn = INTEGRATORS[integrator_name]
            q_next, qdot_next = step_fn(qddot_func, t, q, qdot, dt)

            self._q = q_next
            self._qdot = qdot_next
            self._t = t + dt
            self._last_effort = tau


def _not_positive_number(value: object) -> bool:
    return isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
