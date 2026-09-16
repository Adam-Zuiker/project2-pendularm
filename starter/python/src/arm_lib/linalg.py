"""arm_lib/linalg.py — Generic linear algebra used by the dynamics module:
solving M(q) qddot = b for qddot, plus a handful of plain-list vector
helpers so arm_lib.integrators and arm_lib.dynamics don't need numpy
(the layout doc requires standard-library-only Python).

This module is deliberately free of anything arm-specific: it knows nothing
about joints, links, or gravity. That separation is what lets
arm_lib/dynamics.py's TODO(human) section focus purely on *constructing*
M(q)/C(q,qdot)/G(q) from the Lagrangian, handing off the "now invert/solve
this system" step to code that's already written and tested.
"""

from __future__ import annotations

from typing import List

Vector = List[float]
Matrix = List[List[float]]


def solve_linear_system(a: Matrix, b: Vector) -> Vector:
    """Solve `a @ x = b` for `x` via Gauss-Jordan elimination with partial
    pivoting.

    `a` must be square and non-singular (true for any physically valid
    arm's mass matrix — positive lengths and masses make M(q) symmetric
    positive-definite, per the spec). Works on a copy of `a`/`b`; the
    caller's inputs are left untouched.
    """
    n = len(a)
    if n == 0:
        return []
    if any(len(row) != n for row in a):
        raise ValueError("solve_linear_system: `a` must be square")
    if len(b) != n:
        raise ValueError("solve_linear_system: `b` length must match `a`")

    # Augment [a | b] so every row operation below updates both sides at
    # once; row `i` is `augmented[i][:n]` (a's row) + `augmented[i][n]` (b's
    # entry).
    augmented = [list(a[i]) + [b[i]] for i in range(n)]

    for col in range(n):
        # Partial pivoting: swap in the row with the largest magnitude
        # entry in this column, to keep the elimination numerically stable
        # (dividing by a near-zero pivot would otherwise blow up rounding
        # error).
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot_row][col]) < 1e-12:
            raise ValueError("solve_linear_system: matrix is singular")
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot = augmented[col][col]
        augmented[col] = [value / pivot for value in augmented[col]]

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            augmented[row] = [
                augmented[row][k] - factor * augmented[col][k] for k in range(n + 1)
            ]

    return [augmented[row][n] for row in range(n)]


def vec_add(a: Vector, b: Vector) -> Vector:
    return [ai + bi for ai, bi in zip(a, b)]


def vec_sub(a: Vector, b: Vector) -> Vector:
    return [ai - bi for ai, bi in zip(a, b)]


def vec_scale(a: Vector, s: float) -> Vector:
    return [ai * s for ai in a]
