"""ik_node.py — Registers `/ik/solve`.

Fetches the arm's *current* link lengths fresh on every call via
`ArmRuntime.get_params()` (per the spec: a solve issued after link lengths
changed at runtime should reflect the change) — it never caches them.
Registered the same `(args) -> (values, result, status)` pattern every
other service in this project uses.
"""

from __future__ import annotations

from typing import Any, List, Tuple

from arm_runtime import ArmRuntime
from broker import Broker
from kinematics import solve_ik
from services.provider import InProcessProvider, register as register_provider


def _make_ik_solve_handler(runtime: ArmRuntime):
    def handler(args: Any) -> Tuple[Any, bool, str]:
        if not isinstance(args, dict):
            return {}, False, "args must be an object"

        x = args.get("x")
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return {}, False, "args.x is required and must be numeric"
        y = args.get("y")
        if isinstance(y, bool) or not isinstance(y, (int, float)):
            return {}, False, "args.y is required and must be numeric"
        phi = args.get("phi")
        if phi is not None and (isinstance(phi, bool) or not isinstance(phi, (int, float))):
            return {}, False, "args.phi must be numeric"

        _masses, lengths, _gravity = runtime.get_params()
        try:
            positions = solve_ik(lengths, float(x), float(y), float(phi) if phi is not None else None)
        except ValueError as exc:
            return {}, False, str(exc)
        return {"positions": positions}, True, ""

    return handler


def register(broker: Broker, runtime: ArmRuntime) -> List[InProcessProvider]:
    return [register_provider(broker, "/ik/solve", _make_ik_solve_handler(runtime))]
