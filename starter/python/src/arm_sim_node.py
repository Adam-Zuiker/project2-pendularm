"""arm_sim_node.py — Registers the `/arm_sim/*` services and owns the live
arm's physics loop. `/pid_controller/*` and `/joint_trajectory` land here
too once arm_lib/pid.py's PID controller exists.

`/arm_sim/integration_step` is the standalone, fully-decoupled checkpoint
service (it never reads or writes the live runtime's own `(q, qdot)`, per
the spec) — it doesn't touch `ArmRuntime` at all. Everything else here
does: `set_integrator`/`set_params`/`pause`/`reset` mutate `ArmRuntime`'s
state, and `_PhysicsLoop` is the dedicated thread that actually advances
`(q, qdot)` and publishes `/joint_states`, the same "one dedicated worker
thread" pattern `InProcessProvider` uses for services, just running a loop
instead of waiting on a queue.
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
        steps = args.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise ValueError("args.steps must be an integer")
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
# Physics loop — one dedicated thread that steps ArmRuntime and publishes
# /joint_states, decoupled from any single service call.
# --------------------------------------------------------------------------


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
        while not self._stop_event.is_set():
            dt = self._runtime.dt
            time.sleep(dt)
            if self._stop_event.is_set():
                return
            if not self._runtime.is_paused():
                self._runtime.step()
            self._publish_joint_states()

    def _publish_joint_states(self) -> None:
        t, q, qdot = self._runtime.get_joint_state()
        sec = int(t)
        nanosec = int(round((t - sec) * 1e9))
        msg = {
            "header": {"stamp": {"sec": sec, "nanosec": nanosec}, "frame_id": ""},
            "name": self._runtime.joint_names,
            "position": q,
            "velocity": qdot,
            "effort": [0.0] * len(q),  # real PID effort once pid.py lands
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
    ]
    physics_loop = _PhysicsLoop(broker, runtime)
    physics_loop.start()
    return providers + [physics_loop], runtime
