"""Integrator checks (src/arm_lib/integrators.py and the /arm_sim/integration_step
handler loop), run without the runtime."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from arm_lib.integrators import INTEGRATORS, euler_step, midpoint_step, rk4_step, verlet_step


def _linear_ode(t, q, qdot):
    return [t + 2.0 * q[0] + 3.0 * qdot[0]]


def _check_one_step(step_fn, expected_q, expected_qdot):
    q, qdot = step_fn(_linear_ode, 1.0, [1.0], [2.0], 0.1)
    assert abs(q[0] - expected_q) < 1e-12 and abs(qdot[0] - expected_qdot) < 1e-12, (
        f"{step_fn.__name__} on qddot = t + 2q + 3qdot from t=1, q=1, qdot=2, dt=0.1: "
        f"got q={q}, qdot={qdot}; expected q={expected_q}, qdot={expected_qdot}")


def test_euler_single_step_exact_values():
    _check_one_step(euler_step, 1.2, 2.9)


def test_midpoint_single_step_exact_values():
    _check_one_step(midpoint_step, 1.245, 3.06)


def test_verlet_single_step_exact_values():
    _check_one_step(verlet_step, 1.245, 3.0645)


def test_rk4_single_step_exact_values():
    _check_one_step(rk4_step, 1.0 + 0.1 / 6 * 15.0485, 2.0 + 0.1 / 6 * 64.8415)


def test_integrators_sample_qddot_at_correct_stage_times():
    expected = {
        "euler": [2.0],
        "midpoint": [2.0, 2.125],
        "verlet": [2.0, 2.25],
        "rk4": [2.0, 2.125, 2.125, 2.25],
    }
    for name, times in expected.items():
        seen = []

        def spy(t, q, qdot):
            seen.append(t)
            return [0.0]

        INTEGRATORS[name](spy, 2.0, [0.0], [0.0], 0.25)
        assert len(seen) == len(times) and all(abs(a - b) < 1e-12 for a, b in zip(seen, times)), \
            f"{name} from t=2, dt=0.25 evaluated qddot at times {seen}, expected {times}"


def test_integrators_converge_at_expected_order():
    def forced(t, q, qdot):
        return [-q[0] - qdot[0] - 3.0 * math.sin(2.0 * t) + 2.0 * math.cos(2.0 * t)]

    def errors(step_fn, n, t0=0.3, t_end=1.3):
        dt = (t_end - t0) / n
        t, q, qdot = t0, [math.sin(2.0 * t0)], [2.0 * math.cos(2.0 * t0)]
        for _ in range(n):
            q, qdot = step_fn(forced, t, q, qdot, dt)
            t += dt
        return abs(q[0] - math.sin(2.0 * t_end)), abs(qdot[0] - 2.0 * math.cos(2.0 * t_end))

    bands = {"euler": (1.6, 2.5), "midpoint": (3.2, 5.0), "verlet": (3.2, 5.0), "rk4": (11.0, 40.0)}
    for name, (low, high) in bands.items():
        coarse, fine = errors(INTEGRATORS[name], 20), errors(INTEGRATORS[name], 40)
        for label, c, f in (("position", coarse[0], fine[0]), ("velocity", coarse[1], fine[1])):
            ratio = c / f if f > 0 else float("inf")
            assert low <= ratio <= high, (
                f"{name} {label} error ratio when halving dt is {ratio:.3f} (errors {c:.3e} -> {f:.3e}), "
                f"expected within [{low}, {high}] for its order of accuracy")


def test_integrators_handle_vectors_and_leave_inputs_untouched():
    gains = [1.0, 4.0, 0.25]

    def coupled_free(t, q, qdot):
        return [-k * qi - 0.1 * vi for k, qi, vi in zip(gains, q, qdot)]

    assert set(INTEGRATORS) == {"euler", "midpoint", "verlet", "rk4"}, f"INTEGRATORS keys: {sorted(INTEGRATORS)}"
    for name, fn in (("euler", euler_step), ("midpoint", midpoint_step), ("verlet", verlet_step), ("rk4", rk4_step)):
        assert INTEGRATORS[name] is fn, f"INTEGRATORS[{name!r}] is {INTEGRATORS[name].__name__}, expected {fn.__name__}"
        q0, qdot0 = [0.5, -1.0, 2.0], [0.0, 0.3, -0.7]
        q_in, qdot_in = list(q0), list(qdot0)
        q, qdot = fn(coupled_free, 0.4, q_in, qdot_in, 0.05)
        assert q_in == q0 and qdot_in == qdot0, f"{name} modified its input lists: q={q_in}, qdot={qdot_in}"
        assert len(q) == 3 and len(qdot) == 3, f"{name} returned wrong lengths: q={q}, qdot={qdot}"
        for i, k in enumerate(gains):
            def single(t, qq, vv, k=k):
                return [-k * qq[0] - 0.1 * vv[0]]
            sq, sv = fn(single, 0.4, [q0[i]], [qdot0[i]], 0.05)
            assert abs(q[i] - sq[0]) < 1e-12 and abs(qdot[i] - sv[0]) < 1e-12, (
                f"{name} component {i} of a 3-vector step ({q[i]}, {qdot[i]}) "
                f"differs from the same scalar step ({sq[0]}, {sv[0]})")


def test_integration_step_handler_runs_steps_and_rejects_bad_input():
    from arm_sim_node import _integration_step_handler

    values, ok, message = _integration_step_handler(
        {"function": "t", "x0": 1.0, "xdot0": 2.0, "dt": 0.5, "steps": 2, "integrator": "rk4"})
    assert ok is True, f"valid request rejected: {message}"
    expected = {"times": [0.0, 0.5, 1.0], "positions": [1.0, 2.0 + 0.125 / 6, 3.0 + 1.0 / 6],
                "velocities": [2.0, 2.125, 2.5]}
    for key, want in expected.items():
        got = values[key]
        assert len(got) == len(want) and all(abs(a - b) < 1e-9 for a, b in zip(got, want)), \
            f"rk4 on qddot=t, x0=1, xdot0=2, dt=0.5, steps=2: {key}={got}, expected {want}"

    values, ok, message = _integration_step_handler(
        {"function": "2", "x0": 0.0, "dt": 0.1, "steps": 3.0, "integrator": "euler"})
    assert ok is True and len(values["positions"]) == 4 and values["velocities"][0] == 0.0, \
        f"missing xdot0 should default to 0 and steps=3.0 should be accepted: {values}, {message}"
    assert abs(values["velocities"][-1] - 0.6) < 1e-9, f"euler qddot=2 for 3 steps of 0.1: {values}"

    good = {"function": "t", "x0": 0.0, "xdot0": 0.0, "dt": 0.1, "steps": 1, "integrator": "euler"}
    for change in ({"dt": 0.0}, {"dt": -0.1}, {"steps": 0}, {"steps": -2}, {"steps": 1.5},
                   {"integrator": "bogus"}, {"function": "t+"}, {"function": "foo(t)"}):
        values, ok, message = _integration_step_handler({**good, **change})
        assert ok is False, f"request with {change} should be rejected, got ok={ok}, values={values}"
