# Created by Sami Laubo 05.02.2024

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from numba import njit
import scipy.special

class Diffusion:
    def __init__(self, D, a, b, T, Nx, dt, D_type="constant", D_pos=1, D_neg=1):
        self.D = D
        self.a = a
        self.b = b
        self.T = T
        self.Nx = Nx
        self.dt = dt
        self.dx = (b - a) / Nx
        self.u0 = 1.
        self.D_type = D_type
        self.D_pos = D_pos
        self.D_neg = D_neg
        self.x = np.linspace(self.a, self.b, self.Nx + 1)

        # Create discretized diffusivity
        if D_type == "constant":
            D = np.ones_like(self.x) * self.D
        elif D_type == "step":
            D = np.ones_like(self.x)
            D[self.x < 0] = self.D_neg
            D[self.x >= 0] = self.D_pos
        elif D_type == "x2":
            D = 1 + self.x**2 *10
        elif D_type == "noncont":
            D = D_neg + self.x * D_pos
            D[D<D_neg]= D_neg
        elif D_type == "sin":
            D = D + np.sin(self.x*D_neg + D_pos)
        elif D_type == "stair":
            D = np.zeros_like(self.x) + D_neg
            D += np.floor(self.x-self.a+0.5)%2*D_pos

        self.D_ = D

    def create_matrices_AB(self, reflective=True, plot_diff=False):
        if plot_diff:
            plt.figure()
            plt.plot(self.x, self.D)
            plt.title("Diffusivity")
            plt.show()
            plt.close()

        # Constant diffusivity D = c
        if self.D_type == "constant":
            alpha = self.D_ * self.dt / self.dx**2

            # Matrix for the tridiagonal system
            A = np.zeros((self.Nx + 1, self.Nx + 1))
            np.fill_diagonal(A, 1 + alpha)
            np.fill_diagonal(A[1:], -alpha/2)
            np.fill_diagonal(A[:, 1:], -alpha[1:]/2)

            B = np.zeros((self.Nx + 1, self.Nx + 1))
            np.fill_diagonal(B, 1 - alpha)
            np.fill_diagonal(B[1:], alpha/2)
            np.fill_diagonal(B[:, 1:], alpha[1:]/2)

            # Boundary condition
            if reflective:
                A[0, 1] = -alpha[0]
                A[-1, -2] = -alpha[-1]
                B[0, 1] = alpha[0]
                B[-1, -2] = alpha[-1]

        # Position dependent diffusivity D(x)
        else:
            alpha = self.dt / (2 * self.dx**2)

            D_half_neg = (self.D_[1:] + self.D_[:-1]) / 2
            D_half_pos = (self.D_[:-1] + self.D_[1:]) / 2
            D_half_both = (self.D_[:-2] + 2*self.D_[1:-1] + self.D_[2:]) / 2

            A = np.zeros((self.Nx + 1, self.Nx + 1))
            np.fill_diagonal(A[1:-1,1:-1], 1 + alpha*D_half_both)
            np.fill_diagonal(A[1:], -alpha*D_half_neg)
            np.fill_diagonal(A[:, 1:], -alpha*D_half_pos)

            B = np.zeros((self.Nx + 1, self.Nx + 1))
            np.fill_diagonal(B[1:-1,1:-1], 1 - alpha*D_half_both)
            np.fill_diagonal(B[1:], alpha*D_half_neg)
            np.fill_diagonal(B[:, 1:], alpha*D_half_pos)


        return A, B

    def crank_nicolson_solver(self, reflective=True):
        # Initial condition (Dirac's delta)
        U = np.zeros(self.Nx + 1)
        U[len(U)//2] = 1.0 / self.dx # Approximation of Dirac's delta

        A, B = self.create_matrices_AB(reflective=reflective)

        # Solve over time
        for n in range(int(self.T / self.dt)):
            b = np.dot(B, U)
            U = np.linalg.solve(A, b)

        return self.x, U, self.dx

    # 2.6
    def check_mass_conservation(self, x, y):

        original_mass = 1. # (1 / dx) * dx

        integrated_mass = scipy.integrate.simpson(y, x, self.dx)

        print(f"Original mass: {original_mass} mg")
        print(f"Integrated mass: {integrated_mass} mg")
        print(f'Difference: {abs(original_mass - integrated_mass):e} mg')


    # 2.7
    def analytical_unbounded(self):
        # Start x
        x0 = self.x[(self.Nx+1)//2] # * self.dx

        if self.D_type == "constant":
            u = self.u0 / (np.sqrt(4*np.pi*self.D*self.T)) * np.exp(- (self.x - x0)**2 / (4*self.D*self.T))

        elif self.D_type == "step":
            # A_+(t) (2.10)
            A_pos_1 = scipy.special.erf(x0/(np.sqrt(4*self.D_pos*self.T)))
            A_pos_2 = np.sqrt(self.D_neg/self.D_pos) * np.exp((self.D_pos-self.D_neg) * x0**2 / (4*self.D_pos*self.D_neg*self.T))
            A_pos_3 = scipy.special.erf(x0/(np.sqrt(4*self.D_neg*self.T)))
            A_pos = 2 / (1 + A_pos_1 + A_pos_2 * (1 - A_pos_3))

            # A_-(t) (2.11)
            A_neg = A_pos * A_pos_2

            A = np.ones_like(self.x)
            A[self.x < 0] = A_neg
            A[self.x >= 0] = A_pos

            u = self.u0 * A / np.sqrt(4*np.pi*self.D_*self.T) * np.exp(-(self.x-x0)**2 / (4*self.D_*self.T))


        return self.x, u, self.dx
    
    # 2.8
    def analytical_bounded(self, reflective=True, N=1000):
        # Only for [0, L], but can be translated and dilated from [a, b] - Should be implemented

        @njit
        def u(u0, L, D, t, x, x0, N, reflective):
            total = np.zeros_like(x)

            if reflective:
                total += 1 / L

                for n in range(1, N):
                    total += np.exp(-np.power((n*np.pi/L), 2) * D * t) * 2 / L * np.cos(n*np.pi*x0/L) * np.cos(n*np.pi*x/L)
    
            else: # Absorbing boundary
                for n in range(1, N):
                    total += np.exp(-np.power((n*np.pi/L), 2) * D * t) * 2 / L * np.sin(n*np.pi*x0/L) * np.sin(n*np.pi*x/L)

            return u0 * total
        
        # x = np.linspace(self.a, self.b, self.Nx + 1)
        x = np.linspace(0, self.b - self.a, self.Nx + 1)
        x0 = (self.Nx+1)//2 * self.dx

        return x + self.a, u(self.u0, self.b - self.a, self.D, self.T, x, x0, N, reflective=reflective), self.dx


def main():
    # Example usage
    D = 1.
    D_type = "step"
    a, b = -2.0, 2.0
    T = 0.01
    Nx = 100
    # Nt = 1000
    dt = 1.818e-4


    # Create Diffusion class
    Diff = Diffusion(D, a, b, T, Nx, dt)

    # 2.5
    reflective_CN = Diff.crank_nicolson_solver(D_type=D_type, reflective=True)
    # absorbing_CN = Diff.crank_nicolson_solver(D_type=D_type, reflective=False)

    # 2.6
    Diff.check_mass_conservation(reflective_CN[0], reflective_CN[1])
    # Diff.check_mass_conservation(absorbing_CN[0], absorbing_CN[1])

    # 2.7
    analytical_unbounded = Diff.analytical_unbounded(D_type=D_type)

    # 2.8
    reflective_AB = Diff.analytical_bounded(reflective=True)
    # absorbing_AB = Diff.analytical_bounded(reflective=False)


    # Plotting
    plt.figure()
    plt.title("Reflective Boundaries")
    plt.plot(reflective_CN[0], 2*reflective_CN[1], '.', label='Crank-Nicolson')
    # plt.plot(reflective_AB[0], reflective_AB[1], label="Analytical Solution")
    plt.plot(analytical_unbounded[0], analytical_unbounded[1], label="Analytical Unbounded Solution")
    # plt.title('Crank-Nicolson Solution for Diffusion Equation')
    plt.xlabel('x')
    plt.ylabel('Concentration')
    plt.legend()
    plt.show()

    # plt.figure()
    # plt.title("Absorbing Boundaries")
    # plt.plot(absorbing_CN[0], absorbing_CN[1], '.', label='Crank-Nicolson')
    # plt.plot(absorbing_AB[0], absorbing_AB[1], label="Analytical Solution")
    # plt.xlabel('x')
    # plt.ylabel('Concentration')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()


# Q
    # 2.7
    # Lower mass for step crank-nicolson
