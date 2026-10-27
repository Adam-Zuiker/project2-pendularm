"""Forward/inverse kinematics checks (src/kinematics.py), run without the runtime."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from kinematics import forward_kinematics, solve_ik

TOL = 1e-6


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _assert_reaches(lengths, x, y, phi=None):
    q = solve_ik(lengths, x, y, phi)
    assert len(q) == len(lengths), f"solve_ik{(lengths, x, y, phi)} returned {len(q)} angles: {q}"
    fx, fy, fphi = forward_kinematics(lengths, q)
    assert abs(fx - x) < TOL and abs(fy - y) < TOL, \
        f"solve_ik{(lengths, x, y, phi)} -> q={q}, but FK(q)=({fx}, {fy}) != ({x}, {y})"
    if phi is not None:
        assert abs(_wrap(fphi - phi)) < TOL, \
            f"solve_ik{(lengths, x, y, phi)} -> q={q}, but FK orientation {fphi} != requested phi {phi}"
    return q


def test_forward_kinematics_known_poses():
    cases = [
        ([1.0, 1.0], [0.0, 0.0], (2.0, 0.0, 0.0)),
        ([1.0, 1.0], [math.pi / 2, 0.0], (0.0, 2.0, math.pi / 2)),
        ([2.0, 0.5], [0.0, math.pi / 2], (2.0, 0.5, math.pi / 2)),
        ([1.0, 2.0, 3.0], [0.0, math.pi / 2, math.pi / 2], (-2.0, 2.0, math.pi)),
        ([1.5, 1.0, 0.5], [math.pi / 6, -math.pi / 3, math.pi / 4],
         (1.5 * math.cos(math.pi / 6) + math.cos(-math.pi / 6) + 0.5 * math.cos(math.pi / 12),
          1.5 * math.sin(math.pi / 6) + math.sin(-math.pi / 6) + 0.5 * math.sin(math.pi / 12),
          math.pi / 12)),
    ]
    for lengths, q, expected in cases:
        got = forward_kinematics(lengths, q)
        assert all(abs(g - e) < 1e-9 for g, e in zip(got, expected)), \
            f"forward_kinematics({lengths}, {q}) = {got}, expected {expected}"


def test_ik_two_link_round_trips_through_fk():
    for lengths in ([1.0, 1.0], [1.5, 0.7], [0.6, 1.4]):
        for x, y in [(1.2, 0.5), (-0.9, 0.8), (0.3, -1.1), (-1.0, -0.4), (0.0, 1.3), (1.0, 0.0)]:
            _assert_reaches(lengths, x, y)


def test_ik_three_link_with_phi_round_trips_through_fk():
    lengths = [1.0, 0.8, 0.5]
    for x, y, phi in [(1.2, 0.6, 0.3), (-0.7, 1.0, 2.0), (0.4, -1.3, -1.2), (1.5, 0.0, 0.0), (-1.0, -0.5, math.pi)]:
        _assert_reaches(lengths, x, y, phi)


def test_ik_three_link_without_phi_round_trips_through_fk():
    lengths = [1.0, 0.8, 0.5]
    for x, y in [(1.5, 0.7), (-1.2, 0.9), (0.2, -1.8), (2.0, 0.0), (-0.6, -0.6)]:
        _assert_reaches(lengths, x, y, None)


def test_ik_accepts_exact_boundary_targets():
    _assert_reaches([1.0, 1.0], 2.0, 0.0)
    _assert_reaches([1.5, 0.7], 0.0, 2.2)
    _assert_reaches([1.5, 0.7], 0.8, 0.0)
    _assert_reaches([1.0, 0.8, 0.5], 0.0, 2.3)
    _assert_reaches([1.0, 0.8, 0.5], 2.3, 0.0, 0.0)
    _assert_reaches([1.0, 0.8, 0.5], -0.3, 0.0, math.pi)


def test_ik_rejects_unreachable_targets():
    unreachable = [
        ([1.0, 1.0], 2.5, 0.0, None),
        ([1.0, 1.0], 0.0, -3.0, None),
        ([1.5, 0.7], 0.3, 0.3, None),
        ([1.0, 0.8, 0.5], 2.5, 0.5, None),
        ([1.0, 0.8, 0.5], 2.0, 0.0, math.pi),
        ([1.0, 0.4, 0.5], 0.5, 0.0, 0.0),
    ]
    for lengths, x, y, phi in unreachable:
        try:
            q = solve_ik(lengths, x, y, phi)
        except ValueError:
            continue
        raise AssertionError(f"solve_ik{(lengths, x, y, phi)} should raise ValueError (unreachable) but returned {q}")
