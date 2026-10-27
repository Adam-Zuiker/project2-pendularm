"""Checks arm_lib.dynamics against independently-computed physics rather
than any specific reference formula for M/C/G — per the spec, "no need to
match any particular exact C(q,qdot) matrix... only the observable
coupling behavior matters." Four independent checks:

  1. M(q) is symmetric and positive-definite (physically required for any
     valid arm — positive masses/lengths).
  2. M(q) is consistent with a kinetic energy computed a totally different
     way: per-link translational + rotational KE, with link-COM velocity
     obtained by numerically differentiating COM position (not by using
     M at all).
  3. G(q) matches the numerical gradient of a potential energy computed
     the same independent way (sum of m*g*y_com per link).
  4. A free (tau=0), undamped rollout conserves total mechanical energy —
     this only holds if M, C, and G are all mutually consistent, so it
     exercises the Coriolis term (the one piece the other checks don't
     touch directly) along with inter-joint coupling.

Run with: python3 -m unittest tests.test_dynamics -v
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arm_lib import dynamics, linalg
from arm_lib.integrators import rk4_step

EPS = 1e-6


def link_com_positions(lengths, q):
    """Independent forward kinematics: world (x, y) of each link's center
    of mass. Does not use arm_lib.dynamics at all."""
    n = len(lengths)
    phi = 0.0
    joint_x, joint_y = 0.0, 0.0
    coms = []
    for i in range(n):
        phi += q[i]
        com_x = joint_x + 0.5 * lengths[i] * math.cos(phi)
        com_y = joint_y + 0.5 * lengths[i] * math.sin(phi)
        coms.append((com_x, com_y))
        joint_x += lengths[i] * math.cos(phi)
        joint_y += lengths[i] * math.sin(phi)
    return coms


def potential_energy(masses, lengths, q, gravity):
    coms = link_com_positions(lengths, q)
    return sum(m * gravity * y for m, (_, y) in zip(masses, coms))


def kinetic_energy_independent(masses, lengths, q, qdot):
    """KE via per-link translational + rotational energy, with COM
    velocity from central finite differences on link_com_positions —
    deliberately not going through arm_lib.dynamics.mass_matrix."""
    n = len(lengths)

    def coms_at(scale):
        q_perturbed = [qi + scale * EPS * qdoti for qi, qdoti in zip(q, qdot)]
        return link_com_positions(lengths, q_perturbed)

    coms_plus = coms_at(1.0)
    coms_minus = coms_at(-1.0)

    total = 0.0
    phidot = 0.0
    for i in range(n):
        vx = (coms_plus[i][0] - coms_minus[i][0]) / (2 * EPS)
        vy = (coms_plus[i][1] - coms_minus[i][1]) / (2 * EPS)
        translational = 0.5 * masses[i] * (vx * vx + vy * vy)
        phidot += qdot[i]
        moment_of_inertia = masses[i] * lengths[i] ** 2 / 12.0
        rotational = 0.5 * moment_of_inertia * phidot * phidot
        total += translational + rotational
    return total


def quadratic_form(matrix, vector):
    n = len(vector)
    return sum(vector[i] * matrix[i][j] * vector[j] for i in range(n) for j in range(n))


class TestMassMatrix(unittest.TestCase):
    MASSES = [1.2, 0.8, 1.5]
    LENGTHS = [0.9, 0.6, 1.1]

    def configurations(self):
        rng = random.Random(0)
        for _ in range(5):
            yield [rng.uniform(-math.pi, math.pi) for _ in self.MASSES]

    def test_symmetric_and_positive_definite(self):
        rng = random.Random(1)
        for q in self.configurations():
            m = dynamics.mass_matrix(self.MASSES, self.LENGTHS, q)
            n = len(m)
            for i in range(n):
                for j in range(n):
                    self.assertAlmostEqual(m[i][j], m[j][i], places=9, msg=f"M not symmetric at q={q}")
            for _ in range(5):
                x = [rng.uniform(-2, 2) for _ in range(n)]
                if all(abs(xi) < 1e-9 for xi in x):
                    continue
                self.assertGreater(quadratic_form(m, x), 0.0, msg=f"M not positive-definite at q={q}, x={x}")

    def test_matches_independent_kinetic_energy(self):
        rng = random.Random(2)
        for q in self.configurations():
            qdot = [rng.uniform(-1.5, 1.5) for _ in self.MASSES]
            m = dynamics.mass_matrix(self.MASSES, self.LENGTHS, q)
            ke_from_m = 0.5 * quadratic_form(m, qdot)
            ke_independent = kinetic_energy_independent(self.MASSES, self.LENGTHS, q, qdot)
            self.assertAlmostEqual(ke_from_m, ke_independent, places=4, msg=f"q={q}, qdot={qdot}")


class TestGravityVector(unittest.TestCase):
    MASSES = [1.2, 0.8, 1.5]
    LENGTHS = [0.9, 0.6, 1.1]
    GRAVITY = 9.81

    def test_matches_numerical_gradient_of_potential_energy(self):
        rng = random.Random(3)
        for _ in range(5):
            q = [rng.uniform(-math.pi, math.pi) for _ in self.MASSES]
            g = dynamics.gravity_vector(self.MASSES, self.LENGTHS, q, self.GRAVITY)
            for i in range(len(q)):
                q_plus = list(q)
                q_minus = list(q)
                q_plus[i] += EPS
                q_minus[i] -= EPS
                v_plus = potential_energy(self.MASSES, self.LENGTHS, q_plus, self.GRAVITY)
                v_minus = potential_energy(self.MASSES, self.LENGTHS, q_minus, self.GRAVITY)
                expected = (v_plus - v_minus) / (2 * EPS)
                self.assertAlmostEqual(g[i], expected, places=3, msg=f"q={q}, joint {i}")


class TestCoriolisCoupling(unittest.TestCase):
    def test_joint_velocity_affects_other_joints_acceleration(self):
        """Spec: 'a 3-link arm should couple more strongly than a 2-link
        one' — at minimum, moving one joint should perturb another joint's
        C(q,qdot)@qdot term. Isolates C by holding q fixed (so M and G
        don't change) and comparing two different qdot vectors."""
        for n in (2, 3):
            masses = [1.0] * n
            lengths = [1.0] * n
            q = [0.4] * n
            qdot_zero = [0.0] * n
            qdot_joint0 = [0.0] * n
            qdot_joint0[0] = 2.0

            c_zero = dynamics.coriolis_forces(masses, lengths, q, qdot_zero)
            c_joint0 = dynamics.coriolis_forces(masses, lengths, q, qdot_joint0)

            other_joint_changed = any(
                abs(c_joint0[j] - c_zero[j]) > 1e-9 for j in range(1, n)
            )
            self.assertTrue(
                other_joint_changed,
                msg=f"n={n}: moving joint 0 (qdot={qdot_joint0}) had no effect on any "
                f"other joint's Coriolis term ({c_zero} -> {c_joint0}) — is C actually wired in?",
            )


class TestEnergyConservation(unittest.TestCase):
    def test_free_swing_conserves_energy(self):
        masses = [1.0, 0.8]
        lengths = [1.0, 0.8]
        gravity = 9.81
        q = [1.2, -0.6]
        qdot = [0.0, 0.0]
        dt = 0.001
        steps = 500  # 0.5s

        def qddot_func(t, q_eval, qdot_eval):
            m = dynamics.mass_matrix(masses, lengths, q_eval)
            c = dynamics.coriolis_forces(masses, lengths, q_eval, qdot_eval)
            g = dynamics.gravity_vector(masses, lengths, q_eval, gravity)
            rhs = linalg.vec_sub(linalg.vec_sub([0.0] * len(q_eval), c), g)
            return linalg.solve_linear_system(m, rhs)

        def total_energy(q_eval, qdot_eval):
            m = dynamics.mass_matrix(masses, lengths, q_eval)
            ke = 0.5 * quadratic_form(m, qdot_eval)
            pe = potential_energy(masses, lengths, q_eval, gravity)
            return ke + pe

        e_initial = total_energy(q, qdot)
        t = 0.0
        for _ in range(steps):
            q, qdot = rk4_step(qddot_func, t, q, qdot, dt)
            t += dt
        e_final = total_energy(q, qdot)

        relative_change = abs(e_final - e_initial) / abs(e_initial)
        self.assertLess(
            relative_change, 1e-3,
            msg=f"energy drifted from {e_initial} to {e_final} over a free swing "
            "— M/C/G are likely inconsistent with each other",
        )


if __name__ == "__main__":
    unittest.main()
