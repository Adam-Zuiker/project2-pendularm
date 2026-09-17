"""Checks kinematics.solve_ik against the round-trip property the spec
itself grades on (feeding the result back through forward_kinematics
reproduces the requested target/orientation) plus the documented
reachability rejections — never against one specific closed-form formula
or elbow configuration.

Run with: python3 -m unittest tests.test_kinematics -v
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kinematics import forward_kinematics, solve_ik


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class TestTwoLinkRoundTrip(unittest.TestCase):
    LENGTHS = [1.2, 0.8]

    def test_reachable_random_targets_round_trip(self):
        rng = random.Random(0)
        l1, l2 = self.LENGTHS
        for _ in range(30):
            # Sample a target that's reachable by construction: pick two
            # joint angles, run forward kinematics, and try to solve back
            # to the same (x, y) -- this can never fail by an unlucky draw.
            q1 = rng.uniform(-math.pi, math.pi)
            q2 = rng.uniform(-math.pi, math.pi)
            x, y, _phi = forward_kinematics(self.LENGTHS, [q1, q2])

            positions = solve_ik(self.LENGTHS, x, y, None)
            self.assertEqual(len(positions), 2)
            x2, y2, _phi2 = forward_kinematics(self.LENGTHS, positions)
            self.assertAlmostEqual(x2, x, places=6, msg=f"target=({x},{y}) got={positions}")
            self.assertAlmostEqual(y2, y, places=6, msg=f"target=({x},{y}) got={positions}")

    def test_boundary_fully_extended_is_reachable(self):
        l1, l2 = self.LENGTHS
        x, y = l1 + l2, 0.0
        positions = solve_ik(self.LENGTHS, x, y, None)
        x2, y2, _ = forward_kinematics(self.LENGTHS, positions)
        self.assertAlmostEqual(x2, x, places=6)
        self.assertAlmostEqual(y2, y, places=6)

    def test_boundary_fully_folded_is_reachable(self):
        l1, l2 = self.LENGTHS
        x, y = abs(l1 - l2), 0.0
        positions = solve_ik(self.LENGTHS, x, y, None)
        x2, y2, _ = forward_kinematics(self.LENGTHS, positions)
        self.assertAlmostEqual(x2, x, places=6)
        self.assertAlmostEqual(y2, y, places=6)

    def test_target_beyond_full_extension_is_rejected(self):
        l1, l2 = self.LENGTHS
        with self.assertRaises(ValueError):
            solve_ik(self.LENGTHS, l1 + l2 + 0.5, 0.0, None)

    def test_target_inside_inner_radius_is_rejected(self):
        l1, l2 = self.LENGTHS
        with self.assertRaises(ValueError):
            solve_ik(self.LENGTHS, abs(l1 - l2) * 0.5, 0.0, None)


class TestThreeLinkRoundTrip(unittest.TestCase):
    LENGTHS = [1.0, 0.8, 0.6]

    def test_reachable_random_targets_with_phi_round_trip(self):
        rng = random.Random(1)
        for _ in range(30):
            q = [rng.uniform(-math.pi, math.pi) for _ in self.LENGTHS]
            x, y, phi = forward_kinematics(self.LENGTHS, q)

            positions = solve_ik(self.LENGTHS, x, y, phi)
            self.assertEqual(len(positions), 3)
            x2, y2, phi2 = forward_kinematics(self.LENGTHS, positions)
            self.assertAlmostEqual(x2, x, places=6, msg=f"target=({x},{y},{phi}) got={positions}")
            self.assertAlmostEqual(y2, y, places=6, msg=f"target=({x},{y},{phi}) got={positions}")
            self.assertAlmostEqual(
                _wrap_to_pi(phi2 - phi), 0.0, places=6, msg=f"target phi={phi} got phi={phi2}"
            )

    def test_wrist_point_outside_first_two_links_reach_is_rejected(self):
        # The target itself is reachable on its own (r=2.3 < sum=2.4), but
        # phi=pi drags the wrist point (target - l3 in the phi direction)
        # out to r_wrist=2.9, well past l1+l2=1.8 -- this must be rejected
        # even though the raw target alone would pass the outer-bound check.
        l1, l2, l3 = self.LENGTHS
        x, y, phi = 2.3, 0.0, math.pi
        self.assertLessEqual(math.hypot(x, y), sum(self.LENGTHS))  # sanity: reachable on its own
        with self.assertRaises(ValueError):
            solve_ik(self.LENGTHS, x, y, phi)

    def test_target_beyond_full_extension_is_rejected(self):
        total = sum(self.LENGTHS)
        with self.assertRaises(ValueError):
            solve_ik(self.LENGTHS, total + 0.5, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
