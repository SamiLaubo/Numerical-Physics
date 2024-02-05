# Created by Sami Laubo 05.02.2024

import numpy as np
import matplotlib.pyplot as plt

def crank_nicolson_solver(D, a, b, T, Nx, dt, reflective=True):
    # Parameters
    dx = (b - a) / Nx
    Nt = int(T / dt)
    alpha = D * dt / dx**2

    # Grid
    x_values = np.linspace(a, b, Nx + 1)

    # Initial condition (Dirac's delta)
    u = np.zeros(Nx + 1)
    x0 = (a + b) / 2.0
    # u0[np.abs(x_values - x0) < dx/2] = 1.0 / dx  # Approximation of Dirac delta
    u[np.abs(x_values - x0) < dx/2] = 1.0 / dx  # Approximation of Dirac delta

    # Matrix for the tridiagonal system
    A = np.zeros((Nx + 1, Nx + 1))
    np.fill_diagonal(A, 1 + alpha)
    np.fill_diagonal(A[1:], -alpha/2)
    np.fill_diagonal(A[:, 1:], -alpha/2)

    # Time-stepping loop
    for n in range(Nt):
        b = np.dot(A, u)
        if reflective: # Reflective boundary condition
            b[0] = b[2]
            b[-1] = b[-3]
        else:  # Absorbing boundaries
            b[0] = b[1]
            b[-1] = b[-2]
        u_new = np.linalg.solve(A, b)

        u = u_new

    return x_values, u

# Example usage
D = 1.
a, b = 0.0, 1.0
T = 0.1
Nx = 51
# Nt = 1000
dt = 1.818e-4


reflective_solution = crank_nicolson_solver(D, a, b, T, Nx, dt, reflective=True)
# absorbing_solution = crank_nicolson_solver(D, a, b, T, Nx, dt, reflective=False)

# Plotting
plt.plot(reflective_solution[0], reflective_solution[1], label='Reflective Boundaries')
# plt.plot(absorbing_solution[0], absorbing_solution[1], label='Absorbing Boundaries')
plt.title('Crank-Nicolson Solution for Diffusion Equation')
plt.xlabel('x')
plt.ylabel('Concentration')
plt.legend()
plt.show()
