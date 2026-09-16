"""Checks arm_lib.integrators against a forcing function with a known
closed-form solution, per docs/PROJECT2_PENDULARM.md's own suggested check:
qddot = f(t) = t, starting from rest, integrates to x(t) = t**3 / 6,
xdot(t) = t**2 / 2. Also checks that each method evaluates qddot at the
correct fractional time within a step (freezing t across a step would
otherwise quietly degrade midpoint/rk4 toward euler's accuracy without
raising any error).

Run with: python3 -m unittest tests.test_integrators -v
(from starter/python, with src/ on PYTHONPATH — see the sys.path line
below).
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arm_lib.integrators import euler_step, midpoint_step, rk4_step, verlet_step


def qddot_of_t(t, q, qdot):
    return [t]


def run(step_fn, dt, steps):
    t, q, qdot = 0.0, [0.0], [0.0]
    for _ in range(steps):
        q, qdot = step_fn(qddot_of_t, t, q, qdot, dt)
        t += dt
    return q[0], qdot[0]


class TestIntegrators(unittest.TestCase):
    DT = 0.001
    STEPS = 2000  # t final = 2.0

    def expected(self):
        t_final = self.DT * self.STEPS
        return t_final**3 / 6.0, t_final**2 / 2.0

    def test_euler_converges_loosely(self):
        x_expected, xdot_expected = self.expected()
        x, xdot = run(euler_step, self.DT, self.STEPS)
        self.assertAlmostEqual(x, x_expected, delta=0.05)
        self.assertAlmostEqual(xdot, xdot_expected, delta=0.05)

    def test_midpoint_is_more_accurate_than_euler(self):
        x_expected, _ = self.expected()
        x_euler, _ = run(euler_step, self.DT, self.STEPS)
        x_midpoint, _ = run(midpoint_step, self.DT, self.STEPS)
        self.assertLess(abs(x_midpoint - x_expected), abs(x_euler - x_expected))

    def test_verlet_is_more_accurate_than_euler(self):
        x_expected, _ = self.expected()
        x_euler, _ = run(euler_step, self.DT, self.STEPS)
        x_verlet, _ = run(verlet_step, self.DT, self.STEPS)
        self.assertLess(abs(x_verlet - x_expected), abs(x_euler - x_expected))

    def test_rk4_is_highly_accurate(self):
        x_expected, xdot_expected = self.expected()
        x, xdot = run(rk4_step, self.DT, self.STEPS)
        self.assertAlmostEqual(x, x_expected, delta=1e-6)
        self.assertAlmostEqual(xdot, xdot_expected, delta=1e-6)

    def test_rk4_beats_midpoint_beats_euler(self):
        x_expected, _ = self.expected()
        x_euler, _ = run(euler_step, self.DT, self.STEPS)
        x_midpoint, _ = run(midpoint_step, self.DT, self.STEPS)
        x_rk4, _ = run(rk4_step, self.DT, self.STEPS)
        err_euler = abs(x_euler - x_expected)
        err_midpoint = abs(x_midpoint - x_expected)
        err_rk4 = abs(x_rk4 - x_expected)
        self.assertLess(err_rk4, err_midpoint)
        self.assertLess(err_midpoint, err_euler)


if __name__ == "__main__":
    unittest.main()
