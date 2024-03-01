# Created by Sami Laubo 26.02.2024

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import glob
import time
import matplotlib.animation as animation
from scipy.constants import hbar

class Schrodinger:
    def __init__(self, L, Nx, Nt=1, T=1, m=1) -> None:
        # Values
        self.L = L
        self.T = T
        self.m = m

        self.x0 = L
        self.t0 = 2*m*L**2 / hbar

        self.Nx = Nx
        self.Nt = Nt

        # Create discretization
        self.t, self.dt = np.linspace(0, T, Nt, retstep=True)
        self.x, self.dx = np.linspace(0, L, Nx, retstep=True)

        # Dimentionless t' and x'
        self.t_, self.dt_ = np.linspace(0, T / self.t0, Nt, retstep=True)
        self.x_, self.dx_ = np.linspace(0, 1, Nx, retstep=True)
   
    # Task 2.4
    def eigen(self, Nx=-1, plot=False, save=False):
        # Solve A Psi = lambda Psi

        # Create matrix and step length dx
        if Nx == -1:
            A = np.zeros((self.Nx, self.Nx))
            dx = self.dx_
        else:
            A = np.zeros((Nx, Nx))
            dx = 1 / (Nx-1)

        A /= dx**2
        np.fill_diagonal(A, 2)
        np.fill_diagonal(A[1:], -1)
        np.fill_diagonal(A[:, 1:], -1)

        # Find eigen values and vectors
        # Normalize - Is already normalized
        eig_vals, eig_vecs = np.linalg.eigh(A)
        eig_vals /= dx**2

        # Analytical solution
        n = np.arange(len(eig_vals))+1
        lmbda = (np.pi*n)**2

        if plot:
            # Plot three first eigenfunctions
            plt.figure()
            for i in range(3):
                plt.plot(eig_vecs[:, i], label=f"eval={eig_vals[i]:.2f}")
            plt.show()

            # Task 2.4
            plt.figure()
            plt.plot(n, eig_vals, label="Numerical eigenvalues")
            plt.plot(n, lmbda, label="Analytical eigenvalues")

            plt.title("Eigenvalues")
            plt.xlabel("n")
            plt.ylabel(r"$\lambda_n = \frac{2mL^2}{\hbar^2}E_n$")
            plt.legend()
            plt.show()

        if save:
            np.save(f"output/t24_eigs/eigval_dx_{dx}", eig_vals)
            np.save(f"output/t24_eigs/eigvec_dx_{dx}", eig_vecs)

        return eig_vals, eig_vecs, lmbda

    def eigval_error_dx(self, Nx_low, Nx_high, N, save=False):
        # Arrays to save results
        eigval_error = np.zeros(N)
        Nx = np.linspace(Nx_low, Nx_high, N, dtype=np.int32)

        # Loop through Nx (dx)
        for i in range(N):
            # Get eigenvalues
            eig_vals_num, _, eig_vals_analytical = self.eigen(Nx=Nx[i], save=save, plot=False)

            # Root-Mean-Square-Deviation (RMSE)
            eigval_error[i] = np.sqrt(np.sum((eig_vals_num - eig_vals_analytical)**2) / len(eig_vals_num))

        plt.figure()
        plt.plot(1 / (Nx-1), eigval_error)
        plt.title("Eigenvalue error")
        plt.xlabel(r"$\Delta x$")
        plt.ylabel("RMSE")
        plt.show()

    def load_eigs(self, all=False):
        # Load eigenvalues and vectors from saved files
        eigval_paths = glob.glob("output/t24_eigs/eigval*")

        dx = [float(p.split("_")[-1][:-4]) for p in eigval_paths]

        if all:
            eig_vals = []
            eig_vecs = []

            for i in range(len(dx)):
                eig_vals.append(np.load(eigval_paths[i]))
                eig_vecs.append(np.load(f"output/t24_eigs/eigvec_dx_{dx[i]}.npy"))

        else:
            min_idx = np.argmin(dx)
        
            eig_vals = np.load(eigval_paths[min_idx])
            eig_vecs = np.load(f"output/t24_eigs/eigvec_dx_{dx[min_idx]}.npy")

        return eig_vals, eig_vecs

    # Task 2.7
    def alpha_n(self, Psi_n, Psi_0):
        return np.trapz(Psi_n*Psi_0)
    
    def check_orthogonality(self):
        eig_vals, eig_vecs = self.load_eigs(all=False)
        
        for i in range(min(eig_vecs.shape[1], 20)):
            for j in range(i+1, min(eig_vecs.shape[1], 20)):
                print(f'{self.alpha_n(eig_vecs[i], eig_vecs[j]) = }')

    # Task 2.10
    def init_cond(self, name="psi_1"):
        # Create initial function
        if name == "psi_1":
            self.Psi_0 =  np.sqrt(2) * np.sin(np.pi * self.x_)
        elif name == "delta":
            self.Psi_0 = np.zeros_like(self.x_)
            self.Psi_0[len(self.Psi_0)//2] = 1 / self.dx_


    def evolve(self, plot=True):
        # Load eigenvalues and vectors
        eig_vals, eig_vecs = self.load_eigs()
        alpha = np.zeros(len(eig_vals))

        # Compute alpha_n
        for n in range(len(eig_vals)):
            alpha[n] = self.alpha_n(eig_vecs[:, n], self.Psi_0)

        # Evolve
        # @njit
        def f(Nt, Nx, t_, eig_vals, eig_vecs):
            Psi = np.zeros((Nt, Nx), dtype=np.complex128)
            for idx, t in enumerate(t_):
                Psi[idx] = np.sum(alpha * np.exp(-1j*eig_vals*t) * eig_vecs, axis=1)

                # Normalize
                Psi[idx] /= np.sqrt(np.trapz(Psi[idx]**2))
            
            return Psi
        
        Psi = f(self.Nt, self.Nx, self.t_, eig_vals, eig_vecs)

        # Plot Psi
        if plot:
            plt.figure()
            for i in range(5):
                plt.plot(self.x_, Psi[Psi.shape[0]//5*i], label=f"t={self.t_[Psi.shape[0]//5*i]:.2e}")
            plt.legend()
            plt.show()


if __name__ == '__main__':
    L = 1

    S = Schrodinger(L=L, Nx=1000, Nt=100, T=1e8)

    # Task 2.4
    # S.eigen(plot=True)

    # Task 2.5
    t1 = time.time()
    S.eigval_error_dx(Nx_low=50, Nx_high=1000, N=20, save=True)
    t2 = time.time()
    print(f'Time: {t2 - t1}')

    # Task 2.7
    # S.check_orthogonality()

    ## Task 2.10
    # Set initial condition to first eigen function
    # S.init_cond(name="delta")
    # t1 = time.time()
    # S.evolve()
    # t2 = time.time()
    # print(f'Time: {t2 - t1}')




# TODO:
    # Fix same size for Nx in loaded eigvals and Psi_0: is correct for 1000