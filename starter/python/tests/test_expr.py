"""Expression parser checks (src/arm_lib/expr.py) for /arm_sim/integration_step's
`function` field, run without the runtime."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from arm_lib.expr import ExprError, parse_function


def test_expr_precedence_functions_domain_errors_and_rejections():
    cases = [
        ("1+2*3", 0.0, 7.0), ("(1+2)*3", 0.0, 9.0), ("10-4-3", 0.0, 3.0), ("8/4/2", 0.0, 1.0),
        ("2^3^2", 0.0, 512.0), ("-2^2", 0.0, 4.0), ("2*-3", 0.0, -6.0), ("2*t+1", 3.0, 7.0),
        ("t^2 - 3*t", 4.0, 4.0), ("1.5e1 + 0.25", 0.0, 15.25), ("sin(t)", 0.5, math.sin(0.5)),
        ("cos(t)", 0.5, math.cos(0.5)), ("tan(t)", 0.5, math.tan(0.5)), ("exp(t)", 1.2, math.exp(1.2)),
        ("sqrt(t)", 2.0, math.sqrt(2.0)), ("ln(t)", 2.0, math.log(2.0)), ("abs(t)", -3.5, 3.5),
        ("1/0", 0.0, math.inf), ("-1/0", 0.0, -math.inf),
    ]
    for text, t, expected in cases:
        got = parse_function(text)(t)
        assert got == expected or abs(got - expected) < 1e-12, f"{text!r} at t={t} gave {got}, expected {expected}"
    for text in ("0/0", "sqrt(-1)", "ln(0)", "ln(-2)"):
        got = parse_function(text)(0.0)
        assert math.isnan(got), f"{text!r} should evaluate to nan, got {got}"
    for text in ("", "   ", "foo", "x+1", "(1+2", "1+2)", "1 2", "1+", "sin 1", "sin(1", "2 $ 3", "*3"):
        try:
            parse_function(text)
        except ExprError:
            continue
        raise AssertionError(f"parse_function({text!r}) should raise ExprError")
