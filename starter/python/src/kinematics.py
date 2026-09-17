"""kinematics.py — Forward and inverse kinematics for a planar, serial,
n-link (n=2 or 3) rotational arm, per docs/PROJECT2_PENDULARM.md's
"Inverse kinematics" section. Same joint-angle convention as
arm_lib/dynamics.py (phi_i = q[0] + ... + q[i], 0-indexed) — this module is
purely geometric, no mass/gravity involved at all.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

Vector = List[float]


def forward_kinematics(lengths: Vector, q: Vector) -> Tuple[float, float, float]:
    """Return (x, y, phi_n): the end effector's world position and its
    absolute orientation (the last link's own heading). Given directly by
    the spec's formula: x = sum(l_i*cos(phi_i)), y = sum(l_i*sin(phi_i)),
    phi_n = q[0] + ... + q[-1]."""
    x, y, phi = 0.0, 0.0, 0.0
    for length, qi in zip(lengths, q):
        phi += qi
        x += length * math.cos(phi)
        y += length * math.sin(phi)
    return x, y, phi


# --------------------------------------------------------------------------
# TODO(human): implement solve_ik(lengths, x, y, phi) -> positions.
#
# Raise ValueError with a specific message for an unreachable target (caught
# by ik_node.py's service handler and turned into result:false); return the
# list of n joint angles otherwise. Either elbow configuration is an
# acceptable answer for a reachable target — grading round-trips your
# result through forward_kinematics above, not a specific elbow choice.
#
# 2-link case (Law of Cosines) — this is also the sub-problem the 3-link
# case reduces to below, so it's worth writing as its own reusable step:
#
#   r = sqrt(x**2 + y**2)
#   reject if r > l1+l2 or r < |l1-l2|  (give yourself a small epsilon
#     tolerance on both bounds — floating-point rounding earlier in the
#     computation can put an exactly-reachable boundary target a tiny bit
#     on the wrong side of a strict comparison, and the spec explicitly
#     wants boundary targets — fully extended, or fully folded — treated
#     as reachable, not rejected by that rounding)
#   cos_q2 = (r**2 - l1**2 - l2**2) / (2*l1*l2)
#   clamp cos_q2 to [-1, 1] before calling math.acos — the same rounding
#     can push it a hair outside acos's domain even for a genuinely
#     reachable target
#   q2 = +-math.acos(cos_q2)   (either sign — elbow up/down, both valid)
#   q1 = math.atan2(y, x) - math.atan2(l2*math.sin(q2), l1 + l2*math.cos(q2))
#
# 3-link case (kinematic decoupling) — phi given:
#   wrist_x, wrist_y = x - l3*cos(phi), y - l3*sin(phi)
#   solve q1, q2 for (wrist_x, wrist_y) via the exact 2-link method above,
#     using l1, l2 (NOT l3) — this is exactly the "position that's
#     reachable on its own but whose requested phi pushes the wrist point
#     outside the first two links' reach" rejection case from the spec
#   q3 = phi - q1 - q2   (since phi = q1 + q2 + q3 by definition)
#
# What's genuinely up to you: if phi is omitted on a 3-link request, the
# problem is underdetermined (3 joints, only 2 position constraints) — you
# choose how to resolve the extra degree of freedom. One reasonable
# option: treat it as if q3=0 (link 3 held straight in line with link 2)
# and solve the 2-link sub-problem with an effective second link of length
# l2+l3; any other consistent resolution is equally acceptable.
# --------------------------------------------------------------------------


def solve_ik(lengths: Vector, x: float, y: float, phi: Optional[float]) -> Vector:
    n = len(lengths)
    return two_link_solver(lengths, x, y, phi) if n == 2 else three_link_solver(lengths, x, y, phi)

def two_link_solver(lengths: Vector, x: float, y: float, phi: Optional[float]):
    r = (x**2 + y**2)**0.5
    l1 = lengths[0]
    l2 = lengths[1]
    eps = 0.01

    if r > (l1 + l2 + eps) or r < abs(l1 - l2) - eps: raise ValueError('target not within reach of arm')

    cos_q2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_q2 = max(min(cos_q2, 1), -1)
    q2 = math.acos(cos_q2)
    q1 = math.atan2(y, x) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))

    return [q1, q2]

def three_link_solver(lengths: Vector, x: float, y: float, phi: Optional[float]):
    r = (x**2 + y**2)**0.5
    l1 = lengths[0]
    l2 = lengths[1]
    l3 = lengths[2]
    eps = 0.01

    if phi is None:
        if r > (l1 + l2 + l3 + eps) or r < abs(l1 - l2 - l3) - eps: raise ValueError('target not within reach of arm')

        cos_q2 = (r**2 - l1**2 - (l2 + l3)**2) / (2 * l1 * (l2 + l3))
        cos_q2 = max(min(cos_q2, 1), -1)
        q2 = math.acos(cos_q2)
        q1 = math.atan2(y, x) - math.atan2((l2 + l3) * math.sin(q2), l1 + (l2 + l3) * math.cos(q2))
        q3 = 0

    else:
        x_wrist = x - l3 * math.cos(phi)
        y_wrist = y - l3 * math.sin(phi)
        r_wrist = (x_wrist**2 + y_wrist**2)**0.5

        if r_wrist > (l1 + l2 + eps) or r_wrist < abs(l1 - l2) - eps: raise ValueError('target not within reach of arm')

        cos_q2 = (r_wrist**2 - l1**2 - l2**2)/(2 * l1 * l2)
        cos_q2 = max(min(cos_q2, 1), -1)
        q2 = math.acos(cos_q2)
        q1 = math.atan2(y_wrist, x_wrist) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
        q3 = phi - q1 - q2

    return [q1, q2, q3]