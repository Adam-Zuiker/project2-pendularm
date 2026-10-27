"""Lagrangian dynamics checks (src/arm_lib/dynamics.py, linalg.py), run without the
runtime. Expected values come from textbook closed forms or from energy computed
independently of dynamics.py."""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from arm_lib import dynamics, linalg
from arm_lib.integrators import rk4_step


def _com_positions(lengths, q):
    phi, jx, jy, coms = 0.0, 0.0, 0.0, []
    for length, qi in zip(lengths, q):
        phi += qi
        coms.append((jx + 0.5 * length * math.cos(phi), jy + 0.5 * length * math.sin(phi)))
        jx += length * math.cos(phi)
        jy += length * math.sin(phi)
    return coms


def _independent_energy(masses, lengths, q, qdot, gravity):
    h = 1e-6
    plus = _com_positions(lengths, [qi + h * vi for qi, vi in zip(q, qdot)])
    minus = _com_positions(lengths, [qi - h * vi for qi, vi in zip(q, qdot)])
    kinetic, potential, phidot = 0.0, 0.0, 0.0
    for i, (m, length) in enumerate(zip(masses, lengths)):
        phidot += qdot[i]
        vx = (plus[i][0] - minus[i][0]) / (2 * h)
        vy = (plus[i][1] - minus[i][1]) / (2 * h)
        kinetic += 0.5 * m * (vx * vx + vy * vy) + 0.5 * (m * length * length / 12.0) * phidot * phidot
    for m, (_, y) in zip(masses, _com_positions(lengths, q)):
        potential += m * gravity * y
    return kinetic, potential


def _close(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_mass_matrix_matches_closed_form_for_one_and_two_links():
    for m, length, q in ((2.0, 1.5, 0.3), (0.7, 0.4, -2.0)):
        got = dynamics.mass_matrix([m], [length], [q])
        assert _close(got[0][0], m * length ** 2 / 3.0, 1e-9), f"1-link M({m}, {length}) = {got}, expected m*l^2/3"

    m1, m2, l1, l2 = 1.3, 0.6, 1.1, 0.8
    for q in ([0.0, 0.0], [0.4, 1.2], [-1.0, -2.5], [2.0, math.pi]):
        c2 = math.cos(q[1])
        expected = [
            [m1 * l1 ** 2 / 3 + m2 * (l1 ** 2 + l2 ** 2 / 3 + l1 * l2 * c2), m2 * (l2 ** 2 / 3 + l1 * l2 * c2 / 2)],
            [m2 * (l2 ** 2 / 3 + l1 * l2 * c2 / 2), m2 * l2 ** 2 / 3],
        ]
        got = dynamics.mass_matrix([m1, m2], [l1, l2], q)
        for i in range(2):
            for j in range(2):
                assert _close(got[i][j], expected[i][j], 1e-9), \
                    f"2-link M at q={q}: got {got}, expected {expected} (entry [{i}][{j}])"


def test_mass_matrix_is_symmetric_positive_definite():
    rng = random.Random(511)
    for n in (2, 3, 4):
        for _ in range(5):
            masses = [rng.uniform(0.2, 3.0) for _ in range(n)]
            lengths = [rng.uniform(0.2, 2.0) for _ in range(n)]
            q = [rng.uniform(-math.pi, math.pi) for _ in range(n)]
            M = dynamics.mass_matrix(masses, lengths, q)
            assert len(M) == n and all(len(row) == n for row in M), f"M has wrong shape for n={n}: {M}"
            for i in range(n):
                for j in range(n):
                    assert _close(M[i][j], M[j][i], 1e-9), f"M not symmetric at [{i}][{j}]: {M}"
            L = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1):
                    s = M[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
                    if i == j:
                        assert s > 1e-9, f"M not positive-definite (Cholesky pivot {s}) for {masses}, {lengths}, {q}: {M}"
                        L[i][i] = math.sqrt(s)
                    else:
                        L[i][j] = s / L[j][j]


def test_mass_matrix_matches_independent_kinetic_energy():
    rng = random.Random(42)
    for n in (2, 3):
        for _ in range(5):
            masses = [rng.uniform(0.3, 2.5) for _ in range(n)]
            lengths = [rng.uniform(0.3, 1.8) for _ in range(n)]
            q = [rng.uniform(-math.pi, math.pi) for _ in range(n)]
            qdot = [rng.uniform(-2.0, 2.0) for _ in range(n)]
            M = dynamics.mass_matrix(masses, lengths, q)
            ke_from_m = 0.5 * sum(qdot[i] * M[i][j] * qdot[j] for i in range(n) for j in range(n))
            ke_true, _ = _independent_energy(masses, lengths, q, qdot, 0.0)
            assert _close(ke_from_m, ke_true, 1e-6), \
                f"0.5 qdot^T M qdot = {ke_from_m} but independent KE = {ke_true} for n={n}, q={q}, qdot={qdot}"


def test_gravity_vector_matches_two_link_closed_form():
    m1, m2, l1, l2 = 1.3, 0.6, 1.1, 0.8
    for g in (9.81, 3.7, 0.0):
        for q in ([0.0, 0.0], [0.4, 1.2], [-math.pi / 2, 0.0], [2.0, -0.7]):
            c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
            expected = [g * ((m1 * l1 / 2 + m2 * l1) * c1 + m2 * l2 / 2 * c12), g * m2 * l2 / 2 * c12]
            got = dynamics.gravity_vector([m1, m2], [l1, l2], q, g)
            assert all(abs(a - b) < 1e-9 for a, b in zip(got, expected)) and len(got) == 2, \
                f"G at q={q}, g={g}: got {got}, expected {expected}"


def test_gravity_vector_matches_potential_energy_gradient():
    rng = random.Random(7)
    h = 1e-6
    for _ in range(5):
        n = 3
        masses = [rng.uniform(0.3, 2.5) for _ in range(n)]
        lengths = [rng.uniform(0.3, 1.8) for _ in range(n)]
        q = [rng.uniform(-math.pi, math.pi) for _ in range(n)]
        gravity = rng.uniform(1.0, 12.0)
        got = dynamics.gravity_vector(masses, lengths, q, gravity)
        for i in range(n):
            qp, qm = list(q), list(q)
            qp[i] += h
            qm[i] -= h
            dv = (_independent_energy(masses, lengths, qp, [0.0] * n, gravity)[1]
                  - _independent_energy(masses, lengths, qm, [0.0] * n, gravity)[1]) / (2 * h)
            assert abs(got[i] - dv) < 1e-5, f"G[{i}] = {got[i]} but dV/dq{i} = {dv} for q={q}, G={got}"


def test_coriolis_matches_two_link_closed_form_and_vanishes_at_rest():
    m1, m2, l1, l2 = 1.3, 0.6, 1.1, 0.8
    for q, qdot in (([0.4, 1.2], [0.7, -1.5]), ([-1.0, -2.5], [2.0, 0.3]), ([0.0, 0.9], [-1.1, 1.1])):
        h = m2 * l1 * (l2 / 2) * math.sin(q[1])
        expected = [-h * (2 * qdot[0] * qdot[1] + qdot[1] ** 2), h * qdot[0] ** 2]
        got = dynamics.coriolis_forces([m1, m2], [l1, l2], q, qdot)
        assert all(abs(a - b) < 1e-5 for a, b in zip(got, expected)) and len(got) == 2, \
            f"C(q,qdot)qdot at q={q}, qdot={qdot}: got {got}, expected {expected}"

    masses, lengths, q = [1.0, 0.7, 0.5], [1.0, 0.8, 0.6], [0.3, -0.4, 0.5]
    at_rest = dynamics.coriolis_forces(masses, lengths, q, [0.0, 0.0, 0.0])
    assert all(abs(v) < 1e-9 for v in at_rest), f"Coriolis with qdot=0 should be 0, got {at_rest}"
    qdot = [0.5, -0.2, 1.1]
    once = dynamics.coriolis_forces(masses, lengths, q, qdot)
    twice = dynamics.coriolis_forces(masses, lengths, q, [2 * v for v in qdot])
    assert any(abs(v) > 1e-3 for v in once), f"3-link Coriolis unexpectedly ~0 for qdot={qdot}: {once}"
    assert all(abs(b - 4 * a) < 1e-5 for a, b in zip(once, twice)), \
        f"Coriolis must be quadratic in qdot: C(2qdot)={twice} but 4*C(qdot)={[4 * a for a in once]}"


def test_free_three_link_rollout_conserves_energy():
    masses, lengths, gravity = [1.0, 0.7, 0.5], [1.0, 0.8, 0.6], 9.81

    def qddot(t, q, qdot):
        M = dynamics.mass_matrix(masses, lengths, q)
        c = dynamics.coriolis_forces(masses, lengths, q, qdot)
        g = dynamics.gravity_vector(masses, lengths, q, gravity)
        return linalg.solve_linear_system(M, [-ci - gi for ci, gi in zip(c, g)])

    q, qdot, dt = [0.3, -0.4, 0.5], [0.5, -0.2, 0.1], 0.002
    e0 = sum(_independent_energy(masses, lengths, q, qdot, gravity))
    for k in range(400):
        q, qdot = rk4_step(qddot, k * dt, q, qdot, dt)
    e1 = sum(_independent_energy(masses, lengths, q, qdot, gravity))
    assert abs(q[0] - 0.3) > 0.2, f"arm barely moved under gravity in 0.8 s: q={q}"
    assert abs(e1 - e0) < 1e-4, f"energy drifted from {e0} to {e1} over a free 0.8 s rollout (final q={q})"


def test_solve_linear_system_needs_pivoting_and_leaves_inputs_untouched():
    a = [[0.0, 2.0, 1.0], [1.0, -1.0, 3.0], [4.0, 1.0, -2.0]]
    x_true = [1.5, -2.0, 0.75]
    b = [sum(a[i][j] * x_true[j] for j in range(3)) for i in range(3)]
    a_copy, b_copy = [list(r) for r in a], list(b)
    x = linalg.solve_linear_system(a, b)
    assert all(abs(xi - ti) < 1e-9 for xi, ti in zip(x, x_true)), f"solve_linear_system returned {x}, expected {x_true}"
    assert a == a_copy and b == b_copy, f"solve_linear_system modified its inputs: a={a}, b={b}"
    try:
        x = linalg.solve_linear_system([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])
    except ValueError:
        pass
    else:
        raise AssertionError(f"singular matrix should raise ValueError, got {x}")
    assert linalg.vec_sub([5.0, 1.0], [2.0, 3.0]) == [3.0, -2.0], "vec_sub([5,1],[2,3]) != [3,-2]"
