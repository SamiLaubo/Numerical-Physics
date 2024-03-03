# Created by Sami Laubo 26.02.2024

%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import glob
import time
import matplotlib.animation as animation
from scipy.constants import hbar
import os

import root_finder

class Schrodinger:
    def __init__(self, L=1, Nx=1000, Nt=1, T=1, m=1, pot_type="well", v0=0) -> None:
        # Values
        self.L = L
        self.T = T
        self.m = m
        self.pot_type = pot_type
        self.v0 = v0

        self.x0 = L
        self.t0 = 2*m*L**2 / hbar

        self.Nx = Nx
        self.Nt = Nt

        # Create discretization
        self.discretize_x_t()

        # Discretize potential
        self.discretize_pot()

    def discretize_x_t(self):
        self.t, self.dt = np.linspace(0, self.T, self.Nt, retstep=True)
        self.x, self.dx = np.linspace(0, self.L, self.Nx, retstep=True)

        # Dimentionless t' and x'
        self.t_, self.dt_ = np.linspace(0, self.T / self.t0, self.Nt, retstep=True)
        self.x_, self.dx_ = np.linspace(0, 1, self.Nx, retstep=True)

    def discretize_pot(self):
        # Infinite well with zero pot
        #
        #   |           |
        #   |           |
        #   |___________|
        #
        if self.pot_type == "well":
            self.pot = np.zeros_like(self.x_)
        
        # Infinite well with zero pot and v0 barrier in the middle
        #
        #   |            |
        #   |     __     |
        #   |____|  |____|
        #
        elif self.pot_type == "barrier":
            self.pot = np.zeros_like(self.x_)
            self.pot[len(self.pot)//3:2*len(self.pot)//3] = self.v0

    def update_Nx(self, new_Nx):
        self.Nx = new_Nx
        self.discretize_x_t()
        self.discretize_pot()

   
    # Task 2.4
    def eigen(self, save=False):
        # Solve A Psi = lambda Psi

        # Create matrix
        A = np.zeros((self.Nx, self.Nx))
        np.fill_diagonal(A, -2 - self.dx_**2 * self.pot)
        np.fill_diagonal(A[1:], 1)
        np.fill_diagonal(A[:, 1:], 1)
        A /= -self.dx_**2

        # Find eigen values and vectors
        self.eig_vals, self.eig_vecs = np.linalg.eigh(A)

        # Normalize
        self.eig_vecs /= np.sqrt(np.trapz(self.eig_vecs**2, self.x_, axis=0))

        # Analytical solution
        self.n = np.arange(len(self.eig_vals))+1
        self.lmbda = (np.pi*self.n)**2

        if save:
            np.save(f"output/t24_eigs/eigval_dx_{self.dx_}", self.eig_vals)
            np.save(f"output/t24_eigs/eigvec_dx_{self.dx_}", self.eig_vecs)


    def eigval_error_dx(self, Nx_low, Nx_high, N, save=False):
        # Arrays to save results
        eigval_error = np.zeros(N)
        Nx = np.linspace(Nx_low, Nx_high, N, dtype=np.int32)

        # Loop through Nx (dx)
        for i in range(N):
            # eig_vals_num, _, eig_vals_analytical = self.eigen(Nx=Nx[i], save=save, plot=False)
            # Update Nx and discretizations
            self.update_Nx(Nx[i])

            # Get eigenvalues
            self.eigen(save=save, plot=False)

            # Root-Mean-Square-Deviation (RMSE)
            eigval_error[i] = np.sqrt(np.sum((self.eig_vals - self.lmbda)**2) / len(self.eig_vals))

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
    def init_cond(self, name="psi_1", eigenfunc_idxs=[]):
        # Create initial function
        if name == "psi_1":
            self.Psi_0 =  np.sqrt(2) * np.sin(np.pi * self.x_)
        elif name == "delta":
            self.Psi_0 = np.zeros_like(self.x_)
            self.Psi_0[len(self.Psi_0)//2] = 1 / self.dx_
        elif name == "eigenfuncs":
            self.Psi_0 = np.zeros_like(self.x_)

            for idx in eigenfunc_idxs:
                self.Psi_0 += self.eig_vecs[:, idx]
            # Normalize
            self.Psi_0 /= np.sqrt(len(eigenfunc_idxs))



    def evolve(self, plot=True, animate=False):
        # Load eigenvalues and vectors
        # eig_vals, eig_vecs = self.load_eigs()
        alpha = np.zeros(len(self.eig_vals))

        # Compute alpha_n
        for n in range(len(self.eig_vals)):
            alpha[n] = self.alpha_n(self.eig_vecs[:, n], self.Psi_0)

        # Evolve
        if plot:
            # @njit
            def f(Nt, Nx, t_, eig_vals, eig_vecs):
                Psi = np.zeros((Nt, Nx), dtype=np.complex128)
                for idx, t in enumerate(t_):
                    Psi[idx] = np.sum(alpha * np.exp(-1j*eig_vals*t) * eig_vecs, axis=1)

                    # Normalize
                    Psi[idx] /= np.sqrt(np.trapz(Psi[idx]*np.conj(Psi[idx])))
                
                return Psi
            
            Psi = f(self.Nt, self.Nx, self.t_, self.eig_vals, self.eig_vecs)

            # Plot Psi
            plt.figure()
            for i in range(5):
                Prob_dens = np.conj(Psi[Psi.shape[0]//5*i]) * Psi[Psi.shape[0]//5*i]
                plt.plot(self.x_, Prob_dens, label=f"t={self.t_[Psi.shape[0]//5*i]:.2e}")
                # plt.plot(self.x_, Psi[Psi.shape[0]//5*i], label=f"t={self.t_[Psi.shape[0]//5*i]:.2e}")
            plt.legend()
            plt.show()

        # Animation
        if animate:
            fig, ax = plt.subplots()
            plt.ylim([-0.001, 0.01])

            # Psi_0
            line, = ax.plot(self.x_, self.Psi_0, label=r"t'=0.00s")
            self.plot_insert_potential(fig, ax, pad=0.001)
            legend = plt.legend(loc="upper center")
            plt.title("Probability density\nInitial wavefunction = " + r"$(\Psi_1 + \Psi_2)/\sqrt{2}$")
            plt.xlabel("x'")
            plt.ylabel(r"$|\Psi(x', t')|^2$")

            def anim_func(i):
                # Calculate and normalize
                Psi = np.sum(alpha * np.exp(-1j*self.eig_vals*self.t_[i]) * self.eig_vecs, axis=1)
                Psi /= np.sqrt(np.trapz(Psi*np.conj(Psi)))

                line.set_ydata(np.conj(Psi)*Psi)
                new_label = r"t'=" + f"{self.t_[i]:.2f}s"
                legend.get_texts()[0].set_text(new_label)
                return line,
            
            anim = animation.FuncAnimation(
                fig,
                anim_func,
                len(self.t_),
                interval = 1,
                repeat=False,
                blit=True
            )

            # plt.legend()

            path = "output/t33_prob_dens/test1.gif"
            if os.path.exists(path):
                os.remove(path)
            anim.save(path, fps=60)
            plt.close()

    def plot_eig_values(self, Schrodinger_2=None, n_eig_vecs=4, n_eig_vals=10, plot_vals_n=False):
            
        # Plot eigenfunctions
        fig, ax = plt.subplots()
        cmap = plt.get_cmap("tab10")

        for i in range(n_eig_vecs-1, -1, -1):
            plt.plot(self.x_, self.eig_vecs[:, i], label=f"n = {i+1}", color=cmap(i))

        # Potential
        self.plot_insert_potential(fig, ax)

        # Plot second solution with x dots
        if Schrodinger_2 is not None:
            for i in range(3, -1, -1):
                skip = len(Schrodinger_2.x_)//50
                plt.plot(Schrodinger_2.x_[::skip], Schrodinger_2.eig_vecs[:, i][::skip], 'x', color=cmap(i))
            
        plt.title("Eigenfunctions")
        plt.ylabel("$\Psi(x')$")
        plt.xlabel("x'")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

        # Energy levels
        fig, ax = plt.subplots()
        text_i = []
        for i in range(n_eig_vals):
            plt.hlines(self.eig_vals[i], 0, 1, label=f"eval={self.eig_vals[i]:.2f}")

            # Text
            if i < n_eig_vals - 1 and abs(self.eig_vals[i] - self.eig_vals[i+1]) < 10:
                text_i.append(str(i+1))
            else:
                text_i.append(str(i+1))
                plt.text(1.1, self.eig_vals[i]-5, f"n={','.join(text_i)}")
                text_i = []

        # Potential
        self.plot_insert_potential(fig, ax, true_size=True)
        plt.title("Eigenvalues")
        plt.ylabel(r"$\lambda_n = \frac{2mL^2}{\hbar^2}E_n$")
        plt.show()


        # Task 2.4
        if plot_vals_n:
            plt.figure()
            plt.plot(self.n, self.eig_vals, label="Numerical eigenvalues")
            plt.plot(self.n, self.lmbda, label="Analytical eigenvalues")

            plt.title("Eigenvalues")
            plt.xlabel("n")
            plt.ylabel(r"$\lambda_n = \frac{2mL^2}{\hbar^2}E_n$")
            plt.legend()
            plt.show()

    def plot_Psi_0(self):
        fig, ax = plt.subplots()
        
        plt.plot(self.x_, self.Psi_0, label=r"$\Psi_0$")

        # Potential
        self.plot_insert_potential(fig, ax)
            
        plt.title("Initial wavefunction")
        plt.ylabel("$\Psi(x')$")
        plt.xlabel("x'")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()

    def plot_insert_potential(self, fig, ax, true_size=False, pad=0.0):
        plt.figure(fig)

        if true_size:
            y_max = ax.get_ylim()[1]
            plt.vlines(0, 0, y_max, color='black')
            plt.vlines(1, 0, y_max, color='black')
            plt.plot(self.x_, self.pot, color="black")
        else:
            y_lims = ax.get_ylim()
            plt.vlines(0, y_lims[0] + pad, y_lims[1] - pad, linestyles='--', color='black', label=r"V$^*$")
            plt.vlines(1, y_lims[0] + pad, y_lims[1] - pad, linestyles='--', color='black')
            plt.plot(self.x_, (self.pot/max(np.max(self.pot), 1)*0.5*y_lims[1]) + y_lims[0] + pad, '--', color="black")

    # Task 3.6
    def eigvals_under_barrier(self, v0_low, v0_high, N):
        v0 = np.linspace(v0_low, v0_high, N)
        lmbda_under = np.zeros_like(v0)

        # Find #lambdas under barrier
        for i, v00 in enumerate(v0):
            # S = Schrodinger(pot_type="barrier", v0=v00)
            # S.eigen()
            self.v0 = v00
            self.discretize_pot()
            self.eigen()
            lmbda_under[i] = np.where(self.eig_vals > v00)[0][0]

        # Plot
        plt.figure()
        plt.plot(v0, lmbda_under)
        plt.xlabel(r"$\nu_0 = \frac{2mL^2}{\hbar^2}\cdot V_0$")
        plt.ylabel("Count")
        plt.title("Number of eigenvalues less than barrier height")
        plt.show()

        # Return value where #lmbda shifts to 1
        last_zero = np.where(lmbda_under>0)[0][0]-1
        return v0[last_zero], v0[last_zero+1]




def Task_2():
    # Create class
    S = Schrodinger(L=1, Nx=1000, Nt=100, T=1e8)

    # Task 2.4
    S.eigen(plot=True)
    S.plot_eig_values(plot_vals_n=True)

    # Task 2.5
    t1 = time.time()
    S.eigval_error_dx(Nx_low=50, Nx_high=1000, N=20, save=True)
    t2 = time.time()
    print(f'Time: {t2 - t1}')

    # Task 2.7
    # S.check_orthogonality()

    ## Task 2.10
    # Set initial condition to first eigen function
    S.init_cond(name="delta")
    t1 = time.time()
    S.evolve()
    t2 = time.time()
    print(f'Time: {t2 - t1}')


def Task_3():
    ## Task 3.1
    # Check if barrier potential gives well when v0=0
    # S_well = Schrodinger(L=1, Nx=1000, pot_type="well")
    # S_barrier = Schrodinger(L=1, Nx=1000, pot_type="barrier", v0=0)

    # S_well.eigen()
    # S_barrier.eigen()
    # S_well.plot_eig_values(S_barrier)

    ## Task 3.2
    # With barrier
    # v0 = 1e3
    # S = Schrodinger(L=1, Nx=1000, pot_type="barrier", v0=v0, Nt=100)
    # S.eigen()
    # # S.plot_eig_values(n_eig_vecs=4)

    # # ## Task 3.3
    # t1 = time.time()
    # S.init_cond(name="eigenfuncs", eigenfunc_idxs=[1,2])
    # S.plot_Psi_0()
    # # Update end time
    # S.T = np.pi / (S.eig_vals[2] - S.eig_vals[1]) * S.t0
    # # Discretize t again
    # S.discretize_x_t()
    # # Evolve
    # S.evolve(plot=False, animate=True)
    # t2 = time.time()
    # print(f'Time: {t2 - t1}')

    ## Task 3.4
    # Root finding
    # Plot function
    # root_finder.plot_f(v0, eig_vals=S.eig_vals[:np.where(S.eig_vals>v0)[0][0]])

    ## Task 3.5
    # Find root values
    # root_finder.find_roots(v0, eig_vals=S.eig_vals[:np.where(S.eig_vals>v0)[0][0]])

    ## Task 3.6
    # S = Schrodinger(pot_type="barrier")
    # # S.eigvals_under_barrier(v0_low=0, v0_high=1e5, N=10)

    # # Find value where #lambda 0->1
    # v0_low, v0_high = 22, 23
    # for _ in range(10):
    #     print(f'{v0_low, v0_high = }')
    #     v0_low, v0_high = S.eigvals_under_barrier(v0_low=v0_low, v0_high=v0_high, N=10)

    


if __name__ == '__main__':
    # Task_2()

    Task_3()




# TODO:
    # Fix same size for Nx in loaded eigvals and Psi_0: is correct for 1000
    # Move mains to new py file


# Questions:
    # Task 3.3: Initial cond psi n=1,3?
    # Task 3.3: No tunneling


# Sammenligne res:
    # Task 3.5 - Root values vs eigenvalues