"""arm_sim_node.py — Registers the `/arm_sim/*` and `/pid_controller/*`
services and owns the live arm's physics loop.

For now this only wires up `/arm_sim/integration_step` — the standalone,
fully-decoupled checkpoint service (it never reads or writes the live
runtime's own `(q, qdot)`, per the spec). The live-arm services
(`set_integrator`, `set_params`, `pause`, `reset`, `/pid_controller/*`) and
the physics thread land alongside arm_runtime.py and arm_lib/dynamics.py.

Registered the same way Project 1 registers `/heapify`/`/heap_sort` (see
services/heap_service.py): a plain `(args) -> (values, result, status)`
function wrapped as an `InProcessProvider`.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from arm_lib.expr import ExprError, parse_function
from arm_lib.integrators import INTEGRATORS
from broker import Broker
from services.provider import InProcessProvider, register as register_provider


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


def register(broker: Broker) -> List[InProcessProvider]:
    """Register every `/arm_sim/*` and `/pid_controller/*` service this
    runtime provides. Returns the created providers so main.py can stop()
    them on shutdown, the same pattern services/__init__.py uses in
    Project 1.
    """
    return [
        register_provider(broker, "/arm_sim/integration_step", _integration_step_handler),
    ]
