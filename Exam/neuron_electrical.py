
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, a, b, x0, Nx, T, Nt, **kwargs):
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

        # Physilogical params
        self.lmbda = kwargs.pop("lmbda", 1)
        self.tau = kwargs.pop("lmbda", 1)
        self.g_K = kwargs.pop("g_k", 1)
        self.V_thr = kwargs.pop("V_thr", 1)
        self.gamma = kwargs.pop("gamma", 1)
        self.VN_Na = kwargs.pop("VN_Na", 1)
        self.VN_K = kwargs.pop("VN_K", 1)
        self.V_appl = kwargs.pop("V_appl", 1)
        self.V_mem = kwargs.pop("V_mem", 1)

        print(f'alpha = {self.lmbda**2*self.dt / (self.dx**2 * self.tau)}')

    def create_matrices_AB(self, scheme="Crank-Nicolson", extra_eq=False):
        # Coefficients
        alpha = self.lmbda**2 * self.dt / (self.dx**2 * self.tau)
        beta = self.dt / self.tau

        # Matrix for the tridiagonal system
        A = np.zeros((self.Nx + 1, self.Nx + 1))
        B = np.zeros((self.Nx + 1, self.Nx + 1))

        if scheme.lower() == "explicit euler":
            np.fill_diagonal(A, 1)

            np.fill_diagonal(B, 1 - 2*alpha - beta)
            np.fill_diagonal(B[1:], alpha)
            np.fill_diagonal(B[:, 1:], alpha)

        if scheme.lower() == "implicit euler":
            np.fill_diagonal(A, 1 + 2*alpha)
            np.fill_diagonal(A[1:], -alpha)
            np.fill_diagonal(A[:, 1:], -alpha)

            np.fill_diagonal(B, 1 - beta)

        if scheme.lower() == "crank-nicolson":
            np.fill_diagonal(A, 1 + alpha)
            np.fill_diagonal(A[1:], -alpha/2)
            np.fill_diagonal(A[:, 1:], -alpha/2)

            if extra_eq:
                np.fill_diagonal(B, 1 - alpha)
            else:
                np.fill_diagonal(B, 1 - alpha - beta)
            np.fill_diagonal(B[1:], alpha/2)
            np.fill_diagonal(B[:, 1:], alpha/2)

        return A, B

    def evolve_scheme(self, scheme="Crank-Nicolson", extra_eq=False):
        
        # Matrix to store V for each timestep
        V = np.zeros((self.Nt+1, self.Nx+1))

        # Initial potential
        if extra_eq:
            V[0] = (self.V_appl - self.V_mem) * np.exp(-self.x_x0_2/(2*self.lmbda**2)) + self.V_mem
        else:
            V[0] = np.exp(-self.x_x0_2)
            V[0] = V[0] / np.sum(V[0])

        # Create matrices
        A, B = self.create_matrices_AB(scheme=scheme, extra_eq=extra_eq)
        B_diag_idx = np.diag_indices_from(B)
        B2 = np.zeros_like(B)

        # Solve over time
        if extra_eq:
            beta = self.dt / self.tau
            for i in range(1, self.Nt+1):
                # B2 = beta / self.g_k *
                B2 = B[:]
                B2[B_diag_idx] -= beta * (100/(1+np.exp(self.gamma*(self.V_thr-V[i-1]))) + 1/5) / self.g_K \
                                       * (V[i-1] - self.VN_Na) + beta * ((V[i-1] - self.VN_K))
                b = np.dot(B2, V[i-1])
                V[i] = np.linalg.solve(A, b)
        else:
            for i in range(1, self.Nt+1):
                b = np.dot(B, V[i-1])
                V[i] = np.linalg.solve(A, b)

        return V[1:]

    # 3.5.b
    def cable_analytical(self):
        """Analytical solution of the cable equation
        """
        # Start from t = dt since t=0 -> divide by zero
        V = np.zeros((self.Nt, self.Nx+1))
        for idx, t in enumerate(self.t[1:]):
            V[idx] = 1. / np.sqrt(4*np.pi*t) * np.exp(-self.x_x0_2/(4*t) - t)

        # Normalize
        # V = (V.T / np.sum(V, axis=1)).T
        V = V / np.sum(V[0])

        return V
    
    def plot_evolution(self, V, path="", ax=None, plot_times=None, text_pad=5):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,5))

        if plot_times is None:
            idxs = np.linspace(0, len(V)-1, 4, dtype=int)
        else:
            idxs = []
            for t in plot_times:
                idxs.append((self.t - t).argmin())

        for i in idxs:
            # Plot V
            ax.plot(self.x, V[i], color="k")

            # "Legend"
            x_max = self.x[np.argmax(V[i])]
            y_max = np.max(V[i])

            ax.plot([x_max, x_max+text_pad], [y_max]*2, "--", color="k", linewidth=0.7)
            ax.text(x_max + text_pad,
                    y_max, 
                    f"t = {self.t[i+1]:.2f}s", 
                    horizontalalignment="left",
                    verticalalignment="center")

        ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]+0.0005])
        ax.grid(False)
        ax.set_xlabel(r"x [$m$]")
        ax.set_ylabel(r"V [V]")

        if len(path) > 0:
            fig.savefig(path)
            plt.show()