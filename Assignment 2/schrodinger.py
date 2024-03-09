# Created by Sami Laubo 26.02.2024

import os
import glob

import numpy as np
import scipy
from tqdm import tqdm
from functools import partial
from scipy.constants import hbar
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Schrodinger:
    def __init__(self, L=1, Nx=1002, Nt=1, T=1, m=1, pot_type="well", v0=0, vr=0) -> None:
        # Values
        self.L = L
        self.T = T
        self.m = m
        self.pot_type = pot_type
        self.v0 = v0
        self.vr = vr

        self.x0 = L
        self.t0 = 2*m*L**2 / hbar

        self.Nx = Nx
        self.Nt = Nt

        # Create discretization
        self.discretize_x_t()

        # Discretize potential
        self.discretize_pot()

    def discretize_x_t(self):
        # self.t, self.dt = np.linspace(0, self.T, self.Nt, retstep=True)
        # self.x, self.dx = np.linspace(0, self.L, self.Nx, retstep=True)

        # Dimentionless t' and x'
        # self.t_, self.dt_ = np.linspace(0, self.T / self.t0, self.Nt, retstep=True)
        self.t_, self.dt_ = np.linspace(0, self.T, self.Nt, retstep=True)
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
        #   |              |
        #   |     ____     |
        #   |____| v0 |____|
        #
        elif self.pot_type == "barrier":
            self.pot = np.zeros_like(self.x_)
            self.pot[len(self.pot)//3:2*len(self.pot)//3] = self.v0


        # Infinite well with zero pot and v0 and vr barriers in the middle
        #
        #   |          ____| 
        #   |     ____|    |
        #   |____| v0 | vr |
        #
        elif self.pot_type == "detuning":
            self.pot = np.zeros_like(self.x_)
            self.pot[len(self.pot)//3:2*len(self.pot)//3] = self.v0
            self.pot[2*len(self.pot)//3:] = self.vr
            

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
        # self.eig_vals, self.eig_vecs = np.linalg.eigh(A)
        diag = 2 / self.dx_**2 + self.pot
        offdiag = -np.ones(self.Nx-1) / self.dx_**2
        # diag = -2 / self.dx_**2 * self.pot
        # offdiag = -np.ones(self.Nx-1) / self.dx_**2
        self.eig_vals, self.eig_vecs = scipy.linalg.eigh_tridiagonal(diag, offdiag)

        # Normalize
        self.eig_vecs /= np.sqrt(np.trapz(self.eig_vecs**2, self.x_, axis=0))

        # Analytical solution
        self.n = np.arange(len(self.eig_vals))+1
        self.lmbda = (np.pi*self.n)**2

        if save:
            np.save(f"output/t24_eigs/eigval_dx_{self.dx_}", self.eig_vals)
            np.save(f"output/t24_eigs/eigvec_dx_{self.dx_}", self.eig_vecs)


    def eigvec_error_dx(self, Nx_low, Nx_high, N, num_eigvecs=10, save=False):
        # Arrays to save results
        eigvec_error = np.zeros((num_eigvecs, N))
        Nx = np.linspace(Nx_low, Nx_high, N, dtype=np.int32)

        # Loop through Nx (dx)
        for i in range(N):
            # eig_vals_num, _, eig_vals_analytical = self.eigen(Nx=Nx[i], save=save, plot=False)
            # Update Nx and discretizations
            self.update_Nx(Nx[i])

            # Get eigenvalues
            self.eigen(save=save)

            # Analytical solution
            psi_analytical = self.x_.repeat(num_eigvecs).reshape(len(self.x_), num_eigvecs)
            psi_analytical *= np.pi * (np.arange(num_eigvecs)+1)
            psi_analytical = np.sqrt(2) * np.sin(psi_analytical)

            # Calculate error of probability RMSE
            eigvec_error[:, i] = np.sqrt(np.sum((self.eig_vecs[:, :num_eigvecs]**2 - psi_analytical**2)**2, axis=0)/psi_analytical.shape[0])



        

        plt.figure()
        for i in range(num_eigvecs):
            plt.plot(1 / (Nx-1), eigvec_error[i])
            x_lims = plt.gca().get_xlim()
            x_text = (21*x_lims[1] - x_lims[0]) / 20 
            plt.text(x_text, eigvec_error[i, 0], f"n={i}")

        plt.title("Eigenfunction error")
        plt.xlabel(r"$\Delta x$")
        plt.ylabel("$RMSE$")
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
    def alpha_n(self, Psi_n, Psi_0, x_=None):
        if x_ is not None:
            return np.trapz(Psi_n*Psi_0, x_)
        else:
            return np.trapz(Psi_n*Psi_0, self.x_)
    
    def check_orthogonality(self, num=100, threshold=1e-9):
        eig_vals, eig_vecs = self.load_eigs(all=False)
        x_ = np.linspace(0, 1, len(eig_vals))
        
        all_less = True
        for i in range(min(eig_vecs.shape[1], num)):
            for j in range(i+1, min(eig_vecs.shape[1], num)):
                a_n = self.alpha_n(eig_vecs[:, i], eig_vecs[:, j], x_=x_)

                if a_n > threshold:
                    all_less = False
                    print(f'alpha_n over threshold (1e-10) for i, j = {i}, {j}: {a_n}')

        if all_less:
            print(f"All alpha_n were less than 1e-10 for the {num} lowest eigenvectors")


    # Task 2.10
    def init_cond(self, name="psi_0", eigenfunc_idxs=[]):
        # Create initial function
        if name == "psi_0":
            self.Psi_0 =  np.sqrt(2) * np.sin(np.pi * self.x_)

            self.Psi_0_text = r"$\Psi_0 = \sqrt{2}\sin{\pi x'}$"
            
        elif name == "delta":
            self.Psi_0 = np.zeros_like(self.x_)
            self.Psi_0[len(self.Psi_0)//2] = 1 / self.dx_

            self.Psi_0_text = r"$\Psi_0 = \delta (x' - 1/2)$"

        elif name == "eigenfuncs":
            self.Psi_0 = np.zeros_like(self.x_)

            for idx in eigenfunc_idxs:
                self.Psi_0 += self.eig_vecs[:, idx]
            # Normalize
            self.Psi_0 /= np.sqrt(len(eigenfunc_idxs))

            # Create Psi_0 text
            self.Psi_0_text = r"$\Psi_0 = "
            if len(eigenfunc_idxs) > 1: self.Psi_0_text += "("
            for is_not_first, idx in enumerate(eigenfunc_idxs):
                if is_not_first:
                    self.Psi_0_text += " + "
                self.Psi_0_text += r"\Psi_" + str(idx)
            if len(eigenfunc_idxs) > 1:
                self.Psi_0_text += r")/\sqrt{" + str(len(eigenfunc_idxs)) + r"}$"
            else:
                self.Psi_0_text += r"$"




    def evolve(self, plot=True, animate=False, start_idx_plot=0, path=""):
        # Load eigenvalues and vectors
        # eig_vals, eig_vecs = self.load_eigs()
        alpha = np.zeros(len(self.eig_vals))

        # Compute alpha_n
        for n in range(len(self.eig_vals)):
            alpha[n] = self.alpha_n(self.eig_vecs[:, n], self.Psi_0)

        # Evolve
        if plot:
            # @njit
            def f(Nt, Nx, t_, eig_vals, eig_vecs, x_):
                Psi = np.zeros((Nt, Nx), dtype=np.complex128)
                for idx, t in enumerate(t_):
                    Psi[idx] = np.sum(alpha * np.exp(-1j*eig_vals*t) * eig_vecs, axis=1)

                    # Normalize
                    Psi[idx] /= np.sqrt(np.trapz(np.conj(Psi[idx])*Psi[idx], x_))
                
                return Psi
            
            Psi = f(self.Nt, self.Nx, self.t_, self.eig_vals, self.eig_vecs, self.x_)

            # Plot Psi
            plt.figure()
            plt.title("Probability density\n" + self.Psi_0_text)
            plt.xlabel(r"$x'$")
            plt.ylabel(r"$|\Psi(x', t')|^2$")
            for i in range(start_idx_plot,5):
                Prob_dens = np.conj(Psi[Psi.shape[0]//5*i]) * Psi[Psi.shape[0]//5*i]
                plt.plot(self.x_, Prob_dens, label=f"t={self.t_[Psi.shape[0]//5*i]:.2e}")
                # plt.plot(self.x_, Psi[Psi.shape[0]//5*i], label=f"t={self.t_[Psi.shape[0]//5*i]:.2e}")
            plt.legend()
            plt.show()

        # Animation
        if animate:
            fig, ax = plt.subplots()
            # plt.ylim([-0.001, 0.01])

            # Psi_0
            line, = ax.plot(self.x_, np.conj(self.Psi_0)*self.Psi_0, label=r"t'=0.00s")
            self.plot_insert_potential(fig, ax, pad=0.001)
            legend = plt.legend(loc="upper center")
            plt.title("Probability density\n" + self.Psi_0_text)
            plt.xlabel(r"$x'$")
            plt.ylabel(r"$|\Psi(x', t')|^2$")

            def anim_func(i):
                # Calculate and normalize
                Psi = np.sum(alpha * np.exp(-1j*self.eig_vals*self.t_[i]) * self.eig_vecs, axis=1)
                Psi /= np.sqrt(np.trapz(np.conj(Psi)*Psi, self.x_))

                line.set_ydata(np.conj(Psi)*Psi)
                new_label = r"t'=" + f"{self.t_[i]:.2e}s"
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

            # path = "output/t33_prob_dens/test1.gif"
            if os.path.exists(path):
                os.remove(path)
            anim.save(path, fps=60)
            print(f"Animation saved to {path}")
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
        plt.xlabel("$x'$")
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
            plt.xlabel("$n$")
            plt.ylabel(r"$\lambda_n = \frac{2mL^2}{\hbar^2}E_n$")
            plt.legend()
            plt.show()

    def plot_Psi_0(self):
        fig, ax = plt.subplots()
        
        plt.plot(self.x_, self.Psi_0, label=r"$\Psi_0$")

        # Potential
        self.plot_insert_potential(fig, ax)
            
        plt.title("Initial wavefunction\n" + self.Psi_0_text)
        plt.ylabel("$\Psi(x')$")
        plt.xlabel("$x'$")
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
    def eigvals_under_barrier(self, v0_low, v0_high, N, plot=True):
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
        if plot:
            plt.figure()
            plt.plot(v0, lmbda_under)
            plt.xlabel(r"$\nu_0 = \frac{2mL^2}{\hbar^2}\cdot V_0$")
            plt.ylabel("Count")
            plt.title("Number of eigenvalues less than barrier height")
            plt.show()

        # Return value where #lmbda shifts to 1
        last_zero = np.where(lmbda_under>0)[0][0]-1
        return v0[last_zero], v0[last_zero+1]

    # Task 3.7
    def forward_scheme(self, method="Forward Euler", plot=False, animate=False):
        # Initial condition
        Psi = np.copy(self.Psi_0).astype(np.complex128)

        # Create matrix
        A = np.zeros((self.Nx, self.Nx), dtype=np.complex128)
        np.fill_diagonal(A, -2 - self.dx_**2 * self.pot)
        np.fill_diagonal(A[1:], 1)
        np.fill_diagonal(A[:, 1:], 1)
        A /= -self.dx_**2
        A *= 1j*self.dt_

        # Function for one step
        def psi_step(Psi, A, method, x_):
            if method == "Forward Euler":
                Psi_new = Psi - A @ Psi
            elif method == "Crank Nicolson":
                a = 1 + A/2
                # b = np.dot((1 - A/2), Psi)
                b = Psi - np.dot((A/2), Psi)

                # Solve system
                Psi_new = np.linalg.solve(a, b)

            # Normalize
            Psi_new /= np.sqrt(np.trapz(np.conj(Psi_new)*Psi_new, x_))

            return Psi_new

        if plot:
            # Plot Psi
            fig, ax = plt.subplots()        
            title = "Probability density with " + method
            if self.Psi_0_text is not None:
                title += "\n" + self.Psi_0_text
            plt.title(title)
            plt.ylabel(r"$|\Psi(x', t')|^2$")
            plt.xlabel("x'")

            for i in range(self.Nt-1):
                # Calculate evolution
                Psi = psi_step(Psi, A, method, self.x_)

                # Plot only for five times
                if i % ((self.Nt-1)//4) == 0:
                    Prob_dens = np.conj(Psi) * Psi
                    plt.plot(self.x_, Prob_dens, label=f"t={self.t_[i]:.2e}")

            self.plot_insert_potential(fig, ax, pad=1)
            plt.legend(loc="upper center")
            plt.show()

        if animate:
            yl = {
                "Crank Nicolson": [0, 5],
                "Forward Euler": [0, 30]
            }

            self.animate_evolution(
                partial(psi_step, A=A, method=method, x_=self.x_), 
                path = f"output/t39_prob_dens/{method}.gif",
                method=method,
                y_lims=yl.get(method)
            )

    def animate_evolution(self, psi_func, path, method="", y_lims=[-0.001, 0.01]):
        fig, ax = plt.subplots()
        plt.ylim(y_lims)

        # Psi_0
        line, = ax.plot(self.x_, self.Psi_0, label=r"t'=0.00s")
        self.plot_insert_potential(fig, ax, pad=1)
        legend = plt.legend(loc="upper center")

        # Make title        
        title = "Probability density"
        if len(method):
            title += " with " + method
        if self.Psi_0_text is not None:
            title += "\n" + self.Psi_0_text

        plt.title(title)
        plt.xlabel("x'")
        plt.ylabel(r"$|\Psi(x', t')|^2$")

        self.Psi = np.copy(self.Psi_0).astype(np.complex128)

        def anim_func(i, psi_func):
            # Calculate and normalize
            self.Psi = psi_func(self.Psi)

            line.set_ydata(np.conj(self.Psi)*self.Psi)
            new_label = r"t'=" + f"{self.t_[i]:.2e}s"
            legend.get_texts()[0].set_text(new_label)
            return line,
        
        anim = animation.FuncAnimation(
            fig,
            anim_func,
            len(self.t_),
            interval = 1,
            repeat=False,
            blit=True,
            fargs=(psi_func,)
        )

        if os.path.exists(path):
            os.remove(path)
        if not os.path.exists("/".join(path.split("/")[:-1])):
            os.makedirs("/".join(path.split("/")[:-1]))
        anim.save(path, fps=10)
        print(f"Animation saved to {path}")
        plt.close()


    # Task 4.1
    def detuning_Vr_dependence(self, vr_low, vr_high, N=100):

        eigenvalues_0 = np.zeros(N)
        eigenvalues_1 = np.zeros(N)
        vr = np.linspace(vr_low, vr_high, N)

        for i in tqdm(range(N)):
            # Set potential
            self.vr = vr[i]

            # Discretize potential again
            self.discretize_pot()

            # Get eigenvalues
            self.eigen()

            # Save
            eigenvalues_0[i] = self.eig_vals[0]
            eigenvalues_1[i] = self.eig_vals[1]

        # Plot
        plt.figure()
        plt.title("Two lowest eigenvalues for detuning")
        plt.xlabel(r"$\nu_r=\frac{2mL^2}{\hbar^2}\cdot V$")
        plt.ylabel(r"$\lambda_n = \frac{2mL^2}{\hbar^2}E_n$")

        plt.plot(vr, eigenvalues_0, label=r"$\lambda_0$")
        plt.plot(vr, eigenvalues_1, label=r"$\lambda_1$")

        plt.legend()
        plt.show()

    # Task 4.2
    def tunneling_amplitude(self, vr_low, vr_high, N=100):

        tau = np.zeros(N)
        vr = np.linspace(vr_low, vr_high, N)
        
        H = np.zeros((self.Nx, self.Nx))
        np.fill_diagonal(H[1:], 1)
        np.fill_diagonal(H[:, 1:], 1)
        H /= -self.dx_**2

        self.vr=0
        self.discretize_pot()
        self.eigen()

        for i in tqdm(range(N)):
            # Set potential
            self.vr = vr[i]

            # Discretize potential again
            self.discretize_pot()
            np.fill_diagonal(H, 2/self.dx_**2 + self.pot)

            # Get eigenvalues
            # self.eigen() # eigvals for nu_r=0 only

            # Calculate tau
            tau[i] = scipy.integrate.simpson(self.eig_vecs[:, 0] * (H @ self.eig_vecs[:, 1]), self.x_)

        # Plot
        plt.figure()
        plt.title("Tunneling amplitude")
        plt.xlabel(r"$\nu_r$")
        plt.ylabel(r"$\tau(\nu_r)$")

        plt.plot(vr, tau, label=r"$\tau$")

        # Fit linear regression
        slope, intercept = np.polyfit(vr, tau, 1)
        plt.plot(vr, slope*vr + intercept, '--', label="Linear regression", color="black")

        print(f'tau(vr) = {slope} * vr + {intercept}')

        plt.legend()
        plt.show()

    # Task 4.4
    def Rabi_oscillations(self):

        # Calculate epsilon_0
        self.vr = 0
        self.discretize_pot()
        self.eigen()

        epsilon_0 = self.eig_vals[1] - self.eig_vals[0]
        omega = epsilon_0
        tau = 0.02 * epsilon_0
        f = np.zeros((2, self.Nt), dtype=np.complex128)

        # States
        g0 = np.array([1, 0], dtype=np.complex128).T # Ground state
        e0 = np.array([0, 1], dtype=np.complex128).T # First excited state


        # Matrices
        M = np.ones((2,2), dtype=np.complex128)
        N = np.zeros((2,2), dtype=np.complex128)

        # Helper constants
        N_const = -1j*self.dt_*tau / hbar# * self.t0
        M_const = -N_const / 2
        exp_const = 1j*epsilon_0 / hbar# * self.t0
        sin_const = omega#*self.t0

        print(f'{self.T = }')
        print(f'{N_const = }')
        print(f'{M_const = }')
        print(f'{exp_const = }')
        print(f'{sin_const = }')

        # Set initial condition to ground state
        f[:, 0] = g0 # Since t=0 gives N=0

        # Keep sum
        Nf_sum = np.zeros((2,), dtype=np.complex128)
        sin = 0
        exp_pos = 0
        exp_neg = 0

        for k in range(1, f.shape[1]):
            if k == 1:
                Nf_sum += f[:, 0]

            else:
                # N uses previous sin and exp_
                N[1, 0] = N_const * sin * exp_pos
                N[0, 1] = N_const * sin * exp_neg

                # Add to sum
                Nf_sum += N @ f[:, k-1]

                # print(f'{N @ f[:, k-1] = }')

            # Update values
            sin = np.sin(sin_const * self.t_[k])
            exp_pos = np.exp(exp_const * self.t_[k])
            exp_neg = np.exp(-exp_const * self.t_[k])

            if k == 2 or k == 10:
                print(f'{exp_neg = }')
                print(f'{exp_pos = }')
                print(f'{sin = }')
            
            # Update M
            M[1, 0] = 1 + M_const * sin * exp_pos
            M[0, 1] = 1 + M_const * sin * exp_neg

            print(f'{M = }')


            f[:, k] = np.linalg.solve(M, Nf_sum)

            # Normalize
            f[:, k] /= np.sqrt(np.sum(np.conj(f[:, k])*f[:, k]))

        # Find probabilities for system to be in state g0
        # Normalize
        # f /= np.sqrt(np.sum(np.conj(f)*f, axis=0))
        prob = g0 @ f
        # prob /= np.sqrt(np.sum(np.conj(prob)*prob, axis=0))
        prob = np.conj(prob)*prob

        # Plot
        plt.figure()
        plt.plot(self.t_, prob, label="Numerical")
        # plt.plot(self.t_, np.sin(self.m*self.L**2*tau/hbar**2 * self.t_)**2, '--', label="Analytical")
        plt.plot(self.t_, np.sin(tau/(2*hbar) * self.t_)**2, '--', label="Analytical")

        plt.legend()
        plt.show()
