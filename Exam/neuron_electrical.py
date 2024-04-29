
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, a, b, x0, Nx, T, Nt, lmbda=1, tau=1):
        # Tau and Lambda is set to 1 always
        self.a = a
        self.b = b
        self.x0 = x0
        self.T = T
        self.Nx = Nx
        self.Nt = Nt
        self.x, self.dx = np.linspace(a, b, Nx + 1, retstep=True)
        self.t, self.dt = np.linspace(0, T, Nt + 1, retstep=True)

        self.x_x0_2 = (self.x - x0)**2

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

    # 3.5.b
    def cable_analytical(self):
        """Analytical solution of the cable equation
        """
        # Start from t = dt since t=0 -> divide by zero
        V = np.zeros((self.Nt, self.Nx+1))
        for idx, t in enumerate(self.t[1:]):
            V[idx] = 1. / np.sqrt(4*np.pi*t) * np.exp(-(self.x_x0_2)/(4*t) - t)

        # Normalize
        V = (V.T / np.sum(V, axis=1)).T

        return V
    
    def plot_evolution(self, V, path=""):
        fig = plt.figure(figsize=(8,5))

        idxs = np.linspace(0, len(V)-1, 4, dtype=int)

        for i in idxs:
            plt.plot(self.x, V[i], color="k")

            if i == len(V) - 1:
                plt.text(self.x[np.argmax(V[i])], 
                        np.max(V[i])-0.0004, 
                        f"t = {self.t[i+1]:.0f}s", horizontalalignment="center",
                        fontsize=12)
            else:
                plt.text(self.x[np.argmax(V[i])], 
                        np.max(V[i])+0.0002, 
                        f"t = {self.t[i+1]:.0f}s", horizontalalignment="center",
                        fontsize=12)

        plt.ylim([plt.gca().get_ylim()[0], plt.gca().get_ylim()[1]+0.0005])
        plt.grid(False)
        plt.xlabel(r"x [$m$]")
        plt.ylabel(r"V [V]")
        plt.show()

        if len(path) > 0:
            fig.savefig(path)