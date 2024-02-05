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
    U = np.zeros(Nx + 1)
    U[(Nx+1)//2] = 1.0 / dx # Approximation of Dirac's delta

    # Matrix for the tridiagonal system
    A = np.zeros((Nx + 1, Nx + 1))
    np.fill_diagonal(A, 1 + alpha)
    np.fill_diagonal(A[1:], -alpha/2)
    np.fill_diagonal(A[:, 1:], -alpha/2)

    B = np.zeros((Nx + 1, Nx + 1))
    np.fill_diagonal(B, 1 - alpha)
    np.fill_diagonal(B[1:], alpha/2)
    np.fill_diagonal(B[:, 1:], alpha/2)


    # Boundary condition
    if reflective:
        A[0, 1] = -alpha
        A[-1, -2] = -alpha
        B[0, 1] = alpha
        B[-1, -2] = alpha

    # Solve over time
    for n in range(Nt):
        b = np.dot(B, U)
        U = np.linalg.solve(A, b)

    return x_values, U

# Example usage
D = 1.
a, b = 0.0, 1.0
T = 0.04
Nx = 101
# Nt = 1000
dt = 1.818e-4


reflective_solution = crank_nicolson_solver(D, a, b, T, Nx, dt, reflective=True)
absorbing_solution = crank_nicolson_solver(D, a, b, T, Nx, dt, reflective=False)

# Plotting
plt.plot(reflective_solution[0], reflective_solution[1], label='Reflective Boundaries')
plt.plot(absorbing_solution[0], absorbing_solution[1], label='Absorbing Boundaries')
plt.title('Crank-Nicolson Solution for Diffusion Equation')
plt.xlabel('x')
plt.ylabel('Concentration')
plt.legend()
plt.show()
