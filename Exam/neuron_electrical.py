
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

        # Physological params
        self.lmbda = kwargs.pop("lmbda", 1)
        self.tau = kwargs.pop("lmbda", 1)
        self.g_K = kwargs.pop("g_k", 1)
        self.V_thr = kwargs.pop("V_thr", 1)
        self.gamma = kwargs.pop("gamma", 1)
        self.VN_Na = kwargs.pop("VN_Na", 1)
        self.VN_K = kwargs.pop("VN_K", 1)
        self.V_appl = kwargs.pop("V_appl", 1)
        self.V_mem = kwargs.pop("V_mem", 1)
        self.Na_channel_pos = kwargs.pop("Na_channel_pos", 0.0)

        self.Na_idx = np.argmin(np.abs(self.x - self.x0 - self.Na_channel_pos))

        if self.Na_channel_pos != 0.0:
            print(f'{self.x[self.Na_idx] = }')

        # Print alpha
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

            # Neuman boundary cond
            B[0, 1] = 2*alpha
            B[-1, -2] = 2*alpha

        if scheme.lower() == "implicit euler":
            np.fill_diagonal(A, 1 + 2*alpha)
            np.fill_diagonal(A[1:], -alpha)
            np.fill_diagonal(A[:, 1:], -alpha)

            np.fill_diagonal(B, 1 - beta)

            # Neuman boundary cond
            A[0, 1] = -2*alpha
            A[-1, -2] = -2*alpha

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

            # Neuman boundary cond
            A[0, 1] = -alpha
            A[-1, -2] = -alpha
            B[0, 1] = alpha
            B[-1, -2] = alpha

        return A, B

    def evolve_scheme(self, scheme="Crank-Nicolson", extra_eq=False):
        # Matrix to store V for each timestep
        V = np.zeros((self.Nt+1, self.Nx+1))

        # Initial potential
        if extra_eq:
            V[0] = (self.V_appl - self.V_mem) * np.exp(-self.x_x0_2/(2*self.lmbda**2)) + self.V_mem
        else:
            # V[0] = np.exp(-self.x_x0_2/100)
            # V[0] = V[0] / np.sum(V[0])
            # Same as analytical with t=1
            V[0] = 1. / np.sqrt(4*np.pi) * np.exp(-self.x_x0_2/(4) - 1)

        # Create matrices
        A, B = self.create_matrices_AB(scheme=scheme, extra_eq=extra_eq)

        # Solve over time
        if extra_eq:
            beta = self.dt / self.tau

            for i in range(1, self.Nt+1):
                b = np.dot(B, V[i-1])
                b -= beta * (V[i-1] - self.VN_K)

                if V[i-1,self.Na_idx] > self.V_thr:
                    b[self.Na_idx] -= beta * (100/(1+np.exp(self.gamma*(self.V_thr-V[i-1,self.Na_idx]))) + 1/5) / self.g_K * (V[i-1,self.Na_idx] - self.VN_Na)

                V[i] = np.linalg.solve(A, b)
        else:
            for i in range(1, self.Nt+1):
                b = np.dot(B, V[i-1])
                V[i] = np.linalg.solve(A, b)

        return V

    # 3.5.b
    def cable_analytical(self, threshold=None):
        """Analytical solution of the cable equation
        """
        
        # Store results
        V = np.zeros((self.Nt+1, self.Nx+1))
        
        if threshold is None:
            # Calculate for all times
            for idx, t in enumerate(self.t):
                V[idx] = 1. / np.sqrt(4*np.pi*t) * np.exp(-self.x_x0_2/(4*t) - t)

            return V
        
        # Calculate for all times
        for idx, t in enumerate(self.t):
            V[idx] = 1. / np.sqrt(4*np.pi*t) * np.exp(-self.x_x0_2/(4*t) - t)

            if V[idx, 0] > threshold:
                print(f'{V[idx, 0] = }')
                print(f'V(a,t) > threshold at t = {t = } s ({t-1} s for numerical scheme)')
                break
    
    def plot_evolution(self, V, path="", ax=None, **kwargs):
        """Plot evolution of V
        """
        # Parse args
        plot_N_times = kwargs.pop("plot_N_times", 4)
        text_pad = kwargs.pop("text_pad", 5)
        use_milli = kwargs.pop("use_milli", False)
        plot_idxs = kwargs.pop("plot_idxs", None)
        top_text = kwargs.pop("top_text", False)
        colors = kwargs.pop("colors", False)
        text = kwargs.pop("text", True)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,5))

        if plot_idxs is None:
            plot_idxs = np.linspace(0, len(V)-1, plot_N_times, dtype=int)

        # Plot V
        x_temp = self.x.copy()
        V_temp = V.copy()
        if use_milli:
            x_temp *= 1000.0
            V_temp *= 1000.0

        for i in plot_idxs:
            if colors:
                ax.plot(x_temp, V_temp[i], label=f"t = {self.t[i]:.2f}s")
            else:
                ax.plot(x_temp, V_temp[i], color="k")

            # "Legend"
            x_max = x_temp[np.argmax(V_temp[i])]
            y_max = np.max(V_temp[i])

            if not colors and text:
                if top_text:
                    ax.text(x_max,
                            y_max+text_pad, 
                            f"t = {self.t[i]:.2f}s", 
                            horizontalalignment="center",
                            verticalalignment="center")
                else:
                    ax.plot([x_max, x_max+text_pad], [y_max]*2, "--", color="k", linewidth=0.7)
                    ax.text(x_max + text_pad,
                            y_max, 
                            f"t = {self.t[i]:.2f}s", 
                            horizontalalignment="left",
                            verticalalignment="center")

        if top_text:
            ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]+text_pad])
        ax.grid(False)
        if use_milli:
            ax.set_xlabel(r"x [mm]")
            ax.set_ylabel(r"V [mV]")
        else:
            ax.set_xlabel(r"x [m]")
            ax.set_ylabel(r"V [V]")

        if colors:
            ax.legend()

        if len(path) > 0:
            fig.savefig(path)
            plt.show()