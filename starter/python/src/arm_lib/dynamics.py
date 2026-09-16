"""arm_lib/dynamics.py — Lagrangian dynamics for a planar, serial, n-link
rotational ("RR...R") arm: the mass matrix M(q), the Coriolis/centrifugal
term C(q, qdot) @ qdot, and the gravity load G(q), per
docs/PROJECT2_PENDULARM.md's "Dynamics" section.

Geometry, exactly as the spec defines it: link i is a uniform rigid rod of
length `lengths[i]` and mass `masses[i]`, hinged to the previous link (link
0's hinge is a fixed base at a static frame's origin). `q[i]` is joint i's
own relative rotation; the absolute world orientation of link i is the
running sum `phi_i = q[0] + q[1] + ... + q[i]` (0-indexed here, matching
this file's lists — the spec's `phi_i = q_1 + ... + q_i` is the same sum,
1-indexed). Gravity acts along world `-y` with magnitude `gravity`.

`arm_runtime.py` calls `mass_matrix`/`coriolis_forces`/`gravity_vector`
fresh at every trial state an integrator evaluates (not just the runtime's
own current state) — see that file's `step()` — so these functions must be
pure functions of their arguments, with no hidden dependence on anything
but `masses`/`lengths`/`q`/`qdot`/`gravity`.
"""

from __future__ import annotations

from typing import List

import math

Vector = List[float]
Matrix = List[List[float]]


# --------------------------------------------------------------------------
# TODO(human): implement the three functions below via Lagrangian mechanics.
#
# Derive for a *general* n-link arm — parameterize by n, rather than writing
# one formula for n=2 and a separate one for n=3 — per the spec's explicit
# warning that two special-cased derivations are an easy way to each look
# right in isolation while actually being subtly wrong.
#
# Suggested derivation path:
#   1. Kinetic energy: for each link i, its center of mass sits at the
#      midpoint of the rod. Write the COM's world position as a function of
#      q (using phi_i as defined above), differentiate to get its velocity
#      in terms of qdot, and add the rod's own rotational kinetic energy
#      about its center — a uniform rod of mass m and length l has moment
#      of inertia m*l**2/12 about an axis through its center. Summing every
#      link's translational + rotational kinetic energy gives a quadratic
#      form in qdot: KE = 0.5 * qdot^T M(q) qdot. Reading off M(q) from
#      that quadratic form (the coefficient of qdot_i*qdot_j, doubled off
#      the diagonal) gives you mass_matrix.
#   2. Potential energy: V(q) = sum_i masses[i] * gravity * y_ci(q), where
#      y_ci is that same center-of-mass y-coordinate from step 1.
#      gravity_vector is the gradient: G_i(q) = dV/dq_i.
#   3. Coriolis/centrifugal: the spec says outright that this term "comes
#      from the Christoffel symbols of M(q)" — c_ijk =
#      0.5*(dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i), and
#      (C(q,qdot) @ qdot)_i = sum_j sum_k c_ijk * qdot_j * qdot_k. You can
#      get the partial derivatives of M either by differentiating your
#      closed-form M(q) by hand, or numerically (central finite differences
#      on your own mass_matrix function) — the spec only cares about the
#      observable coupling behavior, not which route you took to get there.
#
# What the grading/tests actually check (none of it requires matching a
# specific reference formula): M(q) is symmetric and positive-definite for
# any physically valid (positive mass/length) configuration; M(q) is
# consistent with an independently-computed kinetic energy for random
# (q, qdot); G(q) matches the numerical gradient of an independently-
# computed potential energy; and a free (tau=0), undamped rollout
# conserves total mechanical energy over a short simulation, which only
# holds if M, C, and G are all mutually consistent.
# --------------------------------------------------------------------------

def mass_matrix(masses: Vector, lengths: Vector, q: Vector) -> Matrix:
    """Return the n x n mass/inertia matrix M(q)."""
    n = len(masses)
    M = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        e_i = [0.0] * n
        e_i[i] = 1.0
        M[i][i] = 2 * _kinetic_energy(masses, lengths, q, e_i)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            e_ij = [0.0] * n
            e_ij[i] = 1.0
            e_ij[j] = 1.0
            M[i][j] = _kinetic_energy(masses, lengths, q, e_ij) - 0.5 * M[i][i] - 0.5 * M[j][j]

    return M


def coriolis_forces(masses: Vector, lengths: Vector, q: Vector, qdot: Vector) -> Vector:
    """Return the n-vector C(q, qdot) @ qdot (not the full C matrix — this
    project only ever needs the product, per the spec's own framing)."""
    n = len(q)
    eps = 1e-6

    # Step 1: dM[k] is the n x n matrix d(M)/d(q_k), for each k in range(n).
    # Fill in _partial_mass_matrix below (central differences on your own
    # mass_matrix, exactly like the gravity gradient trick).
    dM = [_partial_mass_matrix(masses, lengths, q, k, eps) for k in range(n)]

    # Step 2: Christoffel symbols, combined with qdot:
    #   c_ijk = 0.5*(dM[k][i][j] + dM[j][i][k] - dM[i][j][k])
    #   result[i] += c_ijk * qdot[j] * qdot[k]
    result = [0.0] * n
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i] += 0.5 * (dM[k][i][j] + dM[j][i][k] - dM[i][j][k]) * qdot[j] * qdot[k]

    return result


def _partial_mass_matrix(masses: Vector, lengths: Vector, q: Vector, k: int, eps: float) -> Matrix:
    """Return d(M)/d(q_k) via central finite differences on mass_matrix:
    perturb q[k] by +eps and -eps, call mass_matrix at each, subtract the
    two n x n matrices entrywise, divide by 2*eps."""
    n = len(q)
    q_with_qk_plus_eps = list(q)
    q_with_qk_plus_eps[k] += eps
    q_with_qk_minus_eps = list(q)
    q_with_qk_minus_eps[k] -= eps

    M_plus  = mass_matrix(masses, lengths, q_with_qk_plus_eps)
    M_minus = mass_matrix(masses, lengths, q_with_qk_minus_eps)

    dM_dqk = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dM_dqk[i][j] = (M_plus[i][j] - M_minus[i][j]) / (2*eps)

    return dM_dqk


def gravity_vector(masses: Vector, lengths: Vector, q: Vector, gravity: float) -> Vector:
    """Return the n-vector gravity load G(q)."""
    n = len(masses)
    grav_vect = [0.0] * n

    for i in range(n):
        e_i = [0.0] * n
        e_i[i] = 1.0
        kinematics = _system_kinematics(masses, lengths, q, e_i)
        grav_vect[i] = gravity * sum(x * y for x, y in zip(masses, kinematics['y_dot']))

    return grav_vect

def _system_kinematics(masses: Vector, lengths: Vector, q: Vector, e_i: Vector):
    phi = []
    phi_dot = []
    x_dot = []
    y_dot = []

    phi_tot = 0
    phi_dot_tot = 0
    x_dot_sum = 0
    y_dot_sum = 0

    for curr_theta, curr_l, curr_theta_dot in zip(q, lengths, e_i):
        phi_tot += curr_theta
        phi.append(phi_tot)

        phi_dot_tot += curr_theta_dot
        phi_dot.append(phi_dot_tot)

        x_dot_curr = 0.5 * curr_l * -math.sin(phi_tot) * phi_dot_tot
        y_dot_curr = 0.5 * curr_l * math.cos(phi_tot) * phi_dot_tot

        x_dot.append(x_dot_sum + x_dot_curr)
        y_dot.append(y_dot_sum + y_dot_curr)

        x_dot_sum += 2 * x_dot_curr
        y_dot_sum += 2 * y_dot_curr

    return {
        'phi': phi,
        'phi_dot': phi_dot,
        'x_dot': x_dot,
        'y_dot': y_dot,
    }

def _kinetic_energy(masses: Vector, lengths: Vector, q: Vector, e_i: Vector):
    """Return the n x n mass/inertia matrix M(q)."""
    kinematics = _system_kinematics(masses, lengths, q, e_i)

    KE = 0
    for i in range(len(masses)):
        I = masses[i] * lengths[i]**2/12
        KE += 0.5 * masses[i] * (kinematics['x_dot'][i]**2 + kinematics['y_dot'][i]**2) + 0.5 * I * kinematics['phi_dot'][i]**2

    return KE

