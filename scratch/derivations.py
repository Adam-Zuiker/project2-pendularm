from sympy import symbols, sin, cos, expand
from typing import List

Vector = List[float]
Matrix = List[List[float]]


def mass_matrix(masses: Vector, lengths: Vector, q: Vector, q_dot: Vector):
    """Return the n x n mass/inertia matrix M(q)."""
    n = len(masses)

    m = symbols(f'm1:{n+1}')
    l = symbols(f'l1:{n+1}')
    theta = symbols(f'theta1:{n+1}')
    theta_dot = symbols(f'theta_dot1:{n+1}')

    phi = []
    phi_dot = []
    x_dot = []
    y_dot = []

    phi_tot = 0
    phi_dot_tot = 0
    x_dot_sum = 0
    y_dot_sum = 0

    for curr_theta, curr_l, curr_theta_dot in zip(theta, l, theta_dot):
        phi_tot += curr_theta
        phi.append(phi_tot)

        phi_dot_tot += curr_theta_dot
        phi_dot.append(phi_dot_tot)

        x_dot_curr = 0.5 * curr_l * -sin(phi_tot) * phi_dot_tot
        y_dot_curr = 0.5 * curr_l * cos(phi_tot) * phi_dot_tot

        x_dot.append(x_dot_sum + x_dot_curr)
        y_dot.append(y_dot_sum + y_dot_curr)

        x_dot_sum += 2 * x_dot_curr
        y_dot_sum += 2 * y_dot_curr

    KE = 0
    for i in range(n):
        I = m[i] * l[i]**2/12
        KE += 0.5 * m[i] * (x_dot[i]**2 + y_dot[i]**2) + 0.5 * I * phi_dot[i]**2

    KE = expand(KE)
    print(KE)

if __name__ == '__main__':
    mass_matrix([0,0,0], [0,0,0], [0,0,0])





    

