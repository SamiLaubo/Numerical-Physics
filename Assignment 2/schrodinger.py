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
        self.L = L
        self.T = T
        self.m = m

        self.x0 = L
        self.t0 = 2*m*L**2 / hbar

        self.Nx = Nx
        self.Nt = Nt

        self.t, self.dt = np.linspace(0, T, Nt, retstep=True)
        self.x, self.dx = np.linspace(0, L, Nx, retstep=True)

        # t' and x'
        self.t_, self.dt_ = np.linspace(0, T / self.t0, Nt, retstep=True)
        self.x_, self.dx_ = np.linspace(0, 1, Nx, retstep=True)
   
    # Task 2.4
    def eigen(self, dx=-1, plot=False, save=False):
        # Solve A Psi = lambda Psi

        A = np.zeros((self.Nx, self.Nx))
        np.fill_diagonal(A, 2)
        np.fill_diagonal(A[1:], -1)
        np.fill_diagonal(A[:, 1:], -1)

        if dx == -1:
            A /= self.dx_**2
        else:
            A /= dx**2

        # FIX chach wich version of eigh is correct
        eig_vals, eig_vecs = np.linalg.eigh(A)
        n = np.arange(len(eig_vals))+1
        lmbda = (np.pi*n)**2

        if plot:
            # Task 2.4
            plt.figure()
            plt.plot(n, eig_vals, label="Numerical eigenvalues")
            plt.plot(n, lmbda, label="Analytical eigenvalues")

            # FIX axes energy
            plt.title("Eigenvalues")
            plt.xlabel("n")
            plt.ylabel(r"$\lambda_n$")
            plt.legend()
            plt.show()

        if save:
            np.save(f"output/t24_eigs/eigval_dx_{dx}", eig_vals)
            np.save(f"output/t24_eigs/eigvec_dx_{dx}", eig_vecs)

        return eig_vals, eig_vecs, lmbda

    def eigval_error_dx(self, dx_low, dx_high, N, save=False):
        eigval_error = np.zeros(N)
        dx = np.linspace(dx_low, dx_high, N)

        # FIX parallell
        for i in range(N):
            eig_vals_num, _, eig_vals_anal = self.eigen(dx=dx[i], save=save)

            # Sum of Squared Error (SSE)
            eigval_error[i] = np.sum((eig_vals_num - eig_vals_anal)**2) / len(eig_vals_num)

        plt.figure()
        plt.plot(dx, eigval_error)
        # plt.plot(dx, eigval_error)
        plt.title("Eigenvalue error")
        plt.xlabel(r"$\Delta x$")
        plt.ylabel("Error (SSE)")
        plt.show()

    def load_eigs(self):
        eigval_paths = glob.glob("output/t24_eigs/eigval*")

        dx = [float(p.split("_")[-1][:-4]) for p in eigval_paths]

        min_idx = np.argmin(dx)
    
        eig_vals = np.load(eigval_paths[min_idx])
        eig_vecs = np.load(f"output/t24_eigs/eigvec_dx_{dx[min_idx]}.npy")

        return eig_vals, eig_vecs


if __name__ == '__main__':
    L = 1

    S = Schrodinger(L=L, Nx=1000)

    # Task 2.4
    # S.eigen(plot=True)

    # Task 2.5
    # t1 = time.time()
    # S.eigval_error_dx(dx_low=1/50, dx_high=1/1000, N=20, save=True)
    # t2 = time.time()
    # print(f'Time: {t2 - t1}')