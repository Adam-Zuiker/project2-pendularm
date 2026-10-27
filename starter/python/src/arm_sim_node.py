"""arm_sim_node.py — Registers the `/arm_sim/*` and `/pid_controller/*`
services, subscribes to `/joint_trajectory`, and owns the live arm's
physics loop.

`/arm_sim/integration_step` is the standalone, fully-decoupled checkpoint
service (it never reads or writes the live runtime's own `(q, qdot)`, per
the spec) — it doesn't touch `ArmRuntime` at all. Everything else here
does: `set_integrator`/`set_params`/`pause`/`reset`/`pid_controller/*`
mutate `ArmRuntime`'s state, `_JointTrajectorySink` feeds the PID setpoint
in the same way Project 1's `map_store.py` subscribes to `/map` (a plain
`MessageSink`, no socket), and `_PhysicsLoop` is the dedicated thread that
actually advances `(q, qdot)` and publishes `/joint_states`, the same "one
dedicated worker thread" pattern `InProcessProvider` uses for services,
just running a loop instead of waiting on a queue.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, List, Tuple

from arm_lib.expr import ExprError, parse_function
from arm_lib.integrators import INTEGRATORS
from arm_runtime import ArmRuntime
from broker import Broker
from services.provider import InProcessProvider, register as register_provider

log = logging.getLogger("arm_sim_node")


# --------------------------------------------------------------------------
# /arm_sim/integration_step — standalone checkpoint service.
# --------------------------------------------------------------------------


def _require_number(args: dict, key: str, *, default: float | None = None) -> float:
    if key not in args:
        if default is not None:
            return default
        raise ValueError(f"args.{key} is required")
    value = args[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"args.{key} must be numeric")
    return float(value)


def _integration_step_handler(args: Any) -> Tuple[Any, bool, str]:
    if not isinstance(args, dict):
        return {}, False, "args must be an object"

    try:
        function_text = args.get("function")
        if not isinstance(function_text, str):
            raise ValueError("args.function is required and must be a string")
        x0 = _require_number(args, "x0")
        xdot0 = _require_number(args, "xdot0", default=0.0)
        dt = _require_number(args, "dt")
        steps_raw = args.get("steps")
        # Accept a float that represents a whole number (e.g. 5.0) as well
        # as a real int -- JSON itself has no int/float distinction, and a
        # strict `isinstance(steps, int)` check rejects a perfectly valid
        # whole-number count just because some client serialized it with a
        # decimal point. Genuinely fractional values (5.5) are still rejected.
        if isinstance(steps_raw, bool) or not isinstance(steps_raw, (int, float)):
            raise ValueError("args.steps must be an integer")
        if isinstance(steps_raw, float) and not steps_raw.is_integer():
            raise ValueError("args.steps must be an integer")
        steps = int(steps_raw)
        integrator_name = args.get("integrator")
        if not isinstance(integrator_name, str):
            raise ValueError("args.integrator must be a string")
    except ValueError as exc:
        return {}, False, str(exc)

    if dt <= 0:
        return {}, False, "dt must be positive"
    if steps == 0:
        return {}, False, "steps must be nonzero"
    if steps < 0:
        return {}, False, "steps must be positive"
    step_fn = INTEGRATORS.get(integrator_name)
    if step_fn is None:
        return {}, False, f"unknown integrator {integrator_name!r}"

    try:
        f = parse_function(function_text)
    except ExprError as exc:
        return {}, False, f"invalid function: {exc}"

    def qddot_func(t: float, q: List[float], qdot: List[float]) -> List[float]:
        return [f(t)]

    times = [0.0]
    positions = [x0]
    velocities = [xdot0]
    t = 0.0
    q = [x0]
    qdot = [xdot0]
    for _ in range(steps):
        q, qdot = step_fn(qddot_func, t, q, qdot, dt)
        t += dt
        times.append(t)
        positions.append(q[0])
        velocities.append(qdot[0])

    return {"times": times, "positions": positions, "velocities": velocities}, True, ""


# --------------------------------------------------------------------------
# Live-arm simulation-control services — mutate an ArmRuntime.
# --------------------------------------------------------------------------


def _make_set_integrator_handler(runtime: ArmRuntime):
    def handler(args: Any) -> Tuple[Any, bool, str]:
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return {}, False, "args must be an object"
        method = args.get("method")
        timestep = args.get("timestep")
        if method is not None and not isinstance(method, str):
            return {}, False, "method must be a string"
        if timestep is not None and (isinstance(timestep, bool) or not isinstance(timestep, (int, float))):
            return {}, False, "timestep must be numeric"

        current_method, current_dt, errors = runtime.set_integrator(method, timestep)
        return {"method": current_method, "timestep": current_dt}, not errors, "; ".join(errors)

    return handler


def _make_set_params_handler(runtime: ArmRuntime):
    def handler(args: Any) -> Tuple[Any, bool, str]:
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return {}, False, "args must be an object"
        masses = args.get("masses")
        lengths = args.get("lengths")
        gravity = args.get("gravity")
        if masses is not None and not isinstance(masses, list):
            return {}, False, "masses must be an array"
        if lengths is not None and not isinstance(lengths, list):
            return {}, False, "lengths must be an array"
        if gravity is not None and (isinstance(gravity, bool) or not isinstance(gravity, (int, float))):
            return {}, False, "gravity must be numeric"

        current_masses, current_lengths, current_gravity, errors = runtime.set_params(
            masses, lengths, gravity
        )
        values = {"masses": current_masses, "lengths": current_lengths, "gravity": current_gravity}
        return values, not errors, "; ".join(errors)

    return handler


def _make_pause_handler(runtime: ArmRuntime):
    def handler(args: Any) -> Tuple[Any, bool, str]:
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return {}, False, "args must be an object"
        data = args.get("data")
        if data is not None and not isinstance(data, bool):
            return {}, False, "data must be a boolean"
        paused = runtime.set_pause(data)
        return {"data": paused}, True, ""

    return handler


def _make_reset_handler(runtime: ArmRuntime):
    def handler(args: Any) -> Tuple[Any, bool, str]:
        position, velocity = runtime.reset()
        return {"position": position, "velocity": velocity}, True, ""

    return handler


# --------------------------------------------------------------------------
# PID services — mutate ArmRuntime's PidController.
# --------------------------------------------------------------------------


def _make_pid_enable_handler(runtime: ArmRuntime):
    def handler(args: Any) -> Tuple[Any, bool, str]:
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return {}, False, "args must be an object"
        data = args.get("data")
        if data is not None and not isinstance(data, bool):
            return {}, False, "data must be a boolean"
        enabled = runtime.enable_pid(data)
        return {"data": enabled}, True, ""

    return handler


def _make_pid_set_gains_handler(runtime: ArmRuntime):
    def handler(args: Any) -> Tuple[Any, bool, str]:
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return {}, False, "args must be an object"
        kp = args.get("kp")
        ki = args.get("ki")
        kd = args.get("kd")
        for name, value in (("kp", kp), ("ki", ki), ("kd", kd)):
            if value is not None and not isinstance(value, list):
                return {}, False, f"{name} must be an array"

        current_kp, current_ki, current_kd, errors = runtime.set_pid_gains(kp, ki, kd)
        values = {"kp": current_kp, "ki": current_ki, "kd": current_kd}
        return values, not errors, "; ".join(errors)

    return handler


# --------------------------------------------------------------------------
# /joint_trajectory subscriber — step-setpoint servo, not a trajectory
# follower: only the *last* entry of `points` is used, per the spec.
# --------------------------------------------------------------------------


class _JointTrajectorySink:
    id = "arm_sim:joint_trajectory"

    def __init__(self, runtime: ArmRuntime) -> None:
        self._runtime = runtime

    def send(self, message: dict) -> None:
        if message.get("op") != "publish" or message.get("topic") != "/joint_trajectory":
            return
        msg = message.get("msg")
        if not isinstance(msg, dict):
            return
        points = msg.get("points")
        if not isinstance(points, list) or not points:
            return
        last = points[-1]
        if not isinstance(last, dict):
            return

        n = self._runtime.n_links
        positions = _numeric_vector_or_default(last.get("positions"), n)
        velocities = _numeric_vector_or_default(last.get("velocities"), n)
        self._runtime.set_joint_trajectory_setpoint(positions, velocities)


def _numeric_vector_or_default(value: Any, n: int) -> List[float]:
    if (
        isinstance(value, list)
        and len(value) == n
        and all(not isinstance(v, bool) and isinstance(v, (int, float)) for v in value)
    ):
        return [float(v) for v in value]
    return [0.0] * n


# --------------------------------------------------------------------------
# Physics loop — one dedicated thread that steps ArmRuntime and publishes
# /joint_states, decoupled from any single service call.
#
# Stepping tracks wall-clock real time via a standard "fixed timestep with
# accumulator" pattern, rather than either extreme:
#   - Naively sleeping exactly `dt` every tick (the original design) adds
#     dynamics-computation time *on top of* that sleep, so simulated time
#     falls further and further behind real time as `dt` gets small (this
#     project's per-step computation cost is roughly constant regardless
#     of `dt`, dominated by coriolis_forces's finite differences).
#   - Removing throttling entirely and stepping flat-out (tried and
#     reverted) makes the simulation run many times faster than real time
#     whenever computation is cheap (the common case) — which is worse in
#     its own way: any wall-clock delay before a client's first command
#     takes effect (server startup, connection setup) now corresponds to
#     much more free-swing drift than a real-time simulation would show,
#     which is exactly backwards from what callers reasonably expect from
#     "the arm's runtime clock" tracking wall time under normal operation.
# The accumulator splits the difference correctly: when caught up to real
# time (the common case), it waits before stepping again, so it tracks
# ~1x real time exactly like the original design intended. Only when
# genuinely falling behind (computation cost exceeding `dt`) does it skip
# the wait and step back-to-back to catch back up, which is strictly
# better than the original design's behavior in that same situation.
#
# Whenever this loop *does* run several steps back-to-back to catch up, a
# real, nonzero yield every so often is not optional, and `sleep(0)` is
# not good enough: measured directly, a bare sleep(0) yield lets other
# threads calling set_integrator/set_params/enable_pid/etc. (which need
# ArmRuntime's lock) get progressively more starved out the longer this
# loop runs uninterrupted — a from-scratch server startup measured at
# ~83s in one trial, entirely from thread/import starvation under a tight
# sleep(0) loop, no actual hang. A real sleep reliably fixes it.
# --------------------------------------------------------------------------

PUBLISH_RATE_HZ = 60.0
_WAIT_POLL_S = 0.001
_CATCHUP_YIELD_EVERY_N_STEPS = 20
_CATCHUP_YIELD_S = 0.0005
_PAUSED_POLL_INTERVAL_S = 0.05


class _PhysicsLoop:
    def __init__(self, broker: Broker, runtime: ArmRuntime) -> None:
        self.id = "arm_sim:physics_loop"
        self._broker = broker
        self._runtime = runtime
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="physics-loop", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        last_publish = 0.0
        publish_period = 1.0 / PUBLISH_RATE_HZ
        reference_wall = time.monotonic()
        reference_sim = self._runtime.get_joint_state()[0]
        steps_since_yield = 0

        while not self._stop_event.is_set():
            if self._runtime.is_paused():
                self._publish_joint_states()  # keep publishing a frozen snapshot
                self._stop_event.wait(_PAUSED_POLL_INTERVAL_S)
                reference_wall = time.monotonic()
                reference_sim = self._runtime.get_joint_state()[0]
                steps_since_yield = 0
                continue

            wall_elapsed = time.monotonic() - reference_wall
            sim_elapsed = self._runtime.get_joint_state()[0] - reference_sim

            if sim_elapsed < wall_elapsed:
                # Behind real time: catch up, yielding periodically rather
                # than every step, so catch-up throughput isn't dominated
                # by yield overhead when dt is genuinely too fine to keep
                # up with (see module comment above).
                self._runtime.step()
                steps_since_yield += 1
                if steps_since_yield >= _CATCHUP_YIELD_EVERY_N_STEPS:
                    self._stop_event.wait(_CATCHUP_YIELD_S)
                    steps_since_yield = 0
            else:
                # Caught up (the common case): wait before the next step
                # so this tracks ~1x real time instead of running ahead.
                self._stop_event.wait(_WAIT_POLL_S)
                steps_since_yield = 0

            now = time.monotonic()
            if now - last_publish >= publish_period:
                self._publish_joint_states()
                last_publish = now

    def _publish_joint_states(self) -> None:
        t, q, qdot, effort = self._runtime.get_joint_state()
        sec = int(t)
        nanosec = int(round((t - sec) * 1e9))
        msg = {
            "header": {"stamp": {"sec": sec, "nanosec": nanosec}, "frame_id": ""},
            "name": self._runtime.joint_names,
            "position": q,
            "velocity": qdot,
            "effort": effort,
        }
        self._broker.publish("/joint_states", msg)


def register(broker: Broker, n_links: int) -> Tuple[List[Any], ArmRuntime]:
    """Register every `/arm_sim/*` service and start the physics loop.

    Returns `(stoppables, runtime)`: `stoppables` is every object main.py
    should call `.stop()` on during shutdown (the service providers plus
    the physics loop itself — both expose the same `.stop()`/`.id`
    interface, so main.py's shutdown loop doesn't need to know which is
    which). `runtime` is exposed so a future ik_node.py can read current
    link lengths for `/ik/solve`.
    """
    runtime = ArmRuntime(n_links)
    providers = [
        register_provider(broker, "/arm_sim/integration_step", _integration_step_handler),
        register_provider(broker, "/arm_sim/set_integrator", _make_set_integrator_handler(runtime)),
        register_provider(broker, "/arm_sim/set_params", _make_set_params_handler(runtime)),
        register_provider(broker, "/arm_sim/pause", _make_pause_handler(runtime)),
        register_provider(broker, "/arm_sim/reset", _make_reset_handler(runtime)),
        register_provider(broker, "/pid_controller/enable", _make_pid_enable_handler(runtime)),
        register_provider(broker, "/pid_controller/set_gains", _make_pid_set_gains_handler(runtime)),
    ]
    broker.subscribe(_JointTrajectorySink(runtime), "/joint_trajectory")
    physics_loop = _PhysicsLoop(broker, runtime)
    physics_loop.start()
    return providers + [physics_loop], runtime
