"""arm_lib/integrators.py — Four selectable methods for advancing a
second-order system `qddot = f(t, q, qdot)` forward by one timestep `dt`.

This is the *same* abstraction whether it's driving the live n-DOF arm
(`/arm_sim/set_integrator`) or the standalone 1-DOF checkpoint service
(`/arm_sim/integration_step`) — which is why every function here takes
`q`/`qdot` as plain lists (length 1 for the checkpoint, length n for the
live arm) and a `qddot_func(t, q, qdot) -> list[float]` of matching length,
rather than being specialized to either caller.

See docs/PROJECT2_PENDULARM.md's "Numerical integrators" section for the
exact definition of each method. The one detail every method must get
right: evaluate `qddot_func` at the *correct fractional time* for each
sub-stage (`t`, `t + dt/2`, `t + dt`, whichever apply) — freezing `t`
across a step is a subtle bug that only shows up as reduced accuracy on a
genuinely time-varying forcing function, not as a crash.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

from arm_lib.linalg import vec_add, vec_scale

Vector = List[float]
# qddot_func(t, q, qdot) -> qddot, all vectors the same length.
QddotFunc = Callable[[float, Vector, Vector], Vector]


# --------------------------------------------------------------------------
# TODO(human): implement the four integrators below.
#
# Each has the signature `(qddot_func, t, q, qdot, dt) -> (q_next, qdot_next)`
# and advances the system by exactly one step of size `dt`, starting from
# `(q, qdot)` at time `t`. `vec_add(a, b)` and `vec_scale(a, s)` (imported
# above from arm_lib.linalg) do elementwise list addition/scaling, so you
# don't need to write your own loops over vector components.
#
#   - euler_step: forward Euler. One qddot evaluation, at (t, q, qdot).
#         q_next    = q + qdot*dt
#         qdot_next = qdot + qddot*dt
#
#   - midpoint_step: evaluate qddot at the current state, take a trial
#     half-step to estimate the state at t + dt/2, evaluate qddot *there*,
#     then use that midpoint qddot (and the midpoint qdot, for the position
#     update) to take the real full step from the *original* (q, qdot).
#
#   - verlet_step: velocity Verlet. Predict q_next using the current qdot
#     and qddot; provisionally estimate qdot_next using that same qddot;
#     evaluate qddot again at (t + dt, q_next, provisional qdot_next); then
#     finalize qdot_next by averaging the two qddot evaluations. (This is
#     velocity Verlet rather than plain position Verlet specifically
#     because qddot here can depend on qdot, and plain Verlet has nowhere
#     to plug a velocity-dependent force in.)
#
#   - rk4_step: classical RK4. Four qddot evaluations — k1 at (t, q, qdot),
#     k2 and k3 at two different trial states near t + dt/2, k4 at
#     t + dt — blended with the standard 1:2:2:1 weights for both the
#     position and velocity updates.
#
# All four should reduce to the closed-form answer (within integration
# error) for a simple test case like qddot = f(t) = t starting from rest,
# whose exact solution is x(t) = t**3 / 6 — that's a good way to sanity
# check each method by hand once tests/test_integrators.py exists.
# --------------------------------------------------------------------------


def euler_step(
    qddot_func: QddotFunc, t: float, q: Vector, qdot: Vector, dt: float
) -> Tuple[Vector, Vector]:
    qddot = qddot_func(t, q, qdot)
    q_next = vec_add(q, vec_scale(qdot, dt))
    qdot_next = vec_add(qdot, vec_scale(qddot, dt))
    return (q_next, qdot_next)


def midpoint_step(
    qddot_func: QddotFunc, t: float, q: Vector, qdot: Vector, dt: float
) -> Tuple[Vector, Vector]:
    dt_mid = dt * 0.5
    q_mid = vec_add(q, vec_scale(qdot, dt_mid))
    qdot_mid = vec_add(qdot, vec_scale(qddot_func(t, q, qdot), dt_mid))
    qddot_mid = qddot_func(t + dt_mid, q_mid, qdot_mid)

    q_next = vec_add(q, vec_scale(qdot_mid, dt))
    qdot_next = vec_add(qdot, vec_scale(qddot_mid, dt))
    return (q_next, qdot_next)

def verlet_step(
    qddot_func: QddotFunc, t: float, q: Vector, qdot: Vector, dt: float
) -> Tuple[Vector, Vector]:
    qddot = qddot_func(t, q, qdot)
    q_next = vec_add(q, vec_add(vec_scale(qdot, dt), vec_scale(qddot, 0.5 * dt**2)))

    qdot_prov = vec_add(qdot, vec_scale(qddot, dt))
    qddot_dt = qddot_func(t + dt, q_next, qdot_prov)
    qdot_next = vec_add(qdot, vec_scale(vec_add(qddot, qddot_dt), 0.5 * dt))
    return (q_next, qdot_next)

def rk4_step(
    qddot_func: QddotFunc, t: float, q: Vector, qdot: Vector, dt: float
) -> Tuple[Vector, Vector]:
    k1 = qddot_func(t, q, qdot)

    dt_mid = 0.5 * dt
    q_mid = vec_add(q, vec_scale(qdot, dt_mid))
    qdot_mid = vec_add(qdot, vec_scale(k1, dt_mid))
    k2 = qddot_func(t + dt_mid, q_mid, qdot_mid)

    q_mid2 = vec_add(q, vec_scale(qdot_mid, dt_mid))
    qdot_mid2 = vec_add(qdot, vec_scale(k2, dt_mid))
    k3 = qddot_func(t + dt_mid, q_mid2, qdot_mid2)

    q_end = vec_add(q, vec_scale(qdot_mid2, dt))
    qdot_end = vec_add(qdot, vec_scale(k3, dt))
    k4 = qddot_func(t + dt, q_end, qdot_end)

    q_next = vec_add(q, vec_scale(vec_add(qdot, vec_add(vec_scale(qdot_mid , 2), vec_add(vec_scale(qdot_mid2, 2), qdot_end))), dt/6))
    qdot_next = vec_add(qdot, vec_scale(vec_add(k1, vec_add(vec_scale(k2 , 2), vec_add(vec_scale(k3, 2), k4))), dt/6))

    return (q_next, qdot_next)



INTEGRATORS: dict[str, Callable[..., Tuple[Vector, Vector]]] = {
    "euler": euler_step,
    "midpoint": midpoint_step,
    "verlet": verlet_step,
    "rk4": rk4_step,
}
