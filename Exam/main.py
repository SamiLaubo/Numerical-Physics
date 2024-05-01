# Main code that run all tasks

# For VSCode development
%load_ext autoreload
%autoreload 2

# Libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Code with tasks
import neuron_network as nw
from neuron_electrical import Neuron

# Choose tasks to run
TASK_2 = True
TASK_3 = False

# Subtasks (only if super is true)
TASK_22_a = False
TASK_22_b = False
TASK_22_c = False
TASK_22_d = False
TASK_22_e = True
TASK_22_f = False

TASK_35_c = True
TASK_35_d = True
TASK_35_e = True

TASK_37_a = True
TASK_37_b = True
TASK_37_d = True


# Timer class
import time
class Timer:
    def start(self, task="0.0"):
        self.task = task
        self.t1 = time.time()
        print(f"\nTask {self.task}")

    def end(self):
        self.t2 = time.time()
        print(f'Task {self.task} time: {self.t2 - self.t1:.4e}s')

# Plotting setup
plt.style.use('seaborn-v0_8-whitegrid')
fontsize = 12
plt.rcParams.update({
    "font.size": fontsize,
    "legend.frameon": True,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral"
})
from matplotlib import rc
rc('text', usetex=True)

# With Jupyter / IPykernel
%matplotlib inline
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# Tasks for network model
def Task_2():
    timer = Timer()

    if TASK_22_a: 
        timer.start("2.2a")

        T = nw.create_T(N_nodes=21, N_neighbours=2, plot=True, path="output/task_2/t22a_transformation_matrix.pdf")

        timer.end()

    if TASK_22_b: 
        timer.start("2.2b")

        # Create transformation matrix and state vector
        T = nw.create_T(N_nodes=21, N_neighbours=2)
        V = nw.create_V(N=21)

        # Evolve and plot
        V = nw.evolve_VT(V, T, N=5, plot_idx=[0,1,2,3,4], path="output/task_2/t22b_random_evolution.pdf")

        # Gaussian initial wave
        V_gauss = nw.create_V(N=21, type="Gaussian")
        V = nw.evolve_VT(V_gauss, T, N=50, plot_idx=[0,1,2,3,19], path="output/task_2/t22b_gaussian_evolution.pdf")

        timer.end()

    if TASK_22_c: 
        timer.start("2.2c")

        T = nw.create_T(N_nodes=21, N_neighbours=2)
        nw.eigvals(T)

        timer.end()

    if TASK_22_d: 
        timer.start("2.2d")

        # Figure
        fig, axs = plt.subplots(2, 4, sharex=True, figsize=(25,5))
        axs = axs.ravel()

        # Transformation matrix
        T = nw.create_T(N_nodes=21, N_neighbours=2)

        # Gaussian initial wave
        V_gauss = nw.create_V(N=21, type="Gaussian")
        V_gauss = nw.evolve_VT(V_gauss, T, N=100, plot_idx=[99], axs=[axs[0], axs[4]], titles=["Gaussian", "(a)"])

        # Random initial wave
        V_random = nw.create_V(N=21, type="random")
        V_random = nw.evolve_VT(V_random, T, N=100, plot_idx=[99], axs=[axs[1], axs[5]], titles=["Random", "(b)"])

        # Random initial wave
        V_random2 = nw.create_V(N=21, type="random")
        V_random2 = nw.evolve_VT(V_random2, T, N=100, plot_idx=[99], axs=[axs[2], axs[6]], titles=["Random", "(c)"])
        
        # Random initial wave
        V_random3 = nw.create_V(N=21, type="random")
        V_random3 = nw.evolve_VT(V_random3, T, N=100, plot_idx=[99], axs=[axs[3], axs[7]], titles=["Random", "(d)"])
        
        # Plot
        plt.figure(fig)
        plt.tight_layout()
        plt.show()

        fig.savefig("output/task_2/t22d_evolutions_100steps.pdf")

        # Investigate values
        print(f'{V_gauss[-1] = }')
        print(f'{V_random[-1] = }')
        print(f'{V_random2[-1] = }')
        print(f'{V_random3[-1] = }')

        eigvals, eigvecs = nw.eigvals(T, verbal=False)
    
        print(f'{eigvals = }')
        print(f'{eigvecs[:,-1] = }')

        print(f'Angle between V and largest eigvec: {nw.angle(eigvecs[:,-1], V_random)} rad')


        timer.end()

    if TASK_22_e: 
        timer.start("2.2e")

        T_11 = nw.create_T(N_nodes=11, N_neighbours=2)
        T_10 = nw.create_T(N_nodes=10, N_neighbours=2)
        T = nw.join_networks(T_11, T_10, plot=True, path="output/task_2/t22e_T.pdf")

        nw.eigvals(T)
        # np linalg faster because of small matrix - Lanczov will scale better 

        # Evolve random state
        V = nw.create_V(21)
        V = nw.evolve_VT(V, T, N=100, plot_idx=[99], path="output/task_2/t22e_random_evolution.pdf", use_lim=False)

        timer.end()

    if TASK_22_f: 
        timer.start("2.2f")

        # Figure
        fig, axs = plt.subplots(3, 3, sharex=True, figsize=(25,5))
        axs = axs.ravel()

        # Create transformation matrix
        T = nw.create_T(N_nodes=21, N_neighbours=2)

        # Solver to test
        solvers = ["np.linalg.solve", "scipy.linalg.solve", "scipy.sparse.linalg.spsolve"]

        for i, solver in enumerate(solvers):
            # Create initial state
            V = nw.create_V(21, type="inv_step", normalize=True)
            # Evolve
            V = nw.evolve_VT(V, T, forward=False, N=5, plot_idx=[0,4], method=solver, axs=[axs[i], axs[i+3], axs[i+6]], use_lim=False)
            axs[i+6].set_xlabel("Node index\n" + r"$\mathbf{(" + chr(97+i) + r")}$")
            # V = nw.evolve_VT(V, T, forward=False, N=5, method=solver)

            # More accurate timing
            V = nw.evolve_VT(V, T, forward=False, N=100, method=solver)

        # Plot
        plt.figure(fig)
        plt.tight_layout()
        plt.show()

        fig.savefig("output/task_2/t22f_compare_solvers.pdf")

        timer.end()

# Tasks for PDE model
def Task_3():
    timer = Timer()

    if TASK_35_c: 
        timer.start("3.5c")

        # Parameters
        a = 0.0 # [m]
        b = 30.0 # [m]
        x0 = (b-a)/2
        Nx = 300
        Nt = 2000
        T = 2.0 # [s]
        
        # Schemes
        neuron = Neuron(a, b, x0, Nx, T, Nt)
        V_explicit = neuron.evolve_scheme(scheme="explicit euler")
        V_implicit = neuron.evolve_scheme(scheme="implicit euler")
        V_crank = neuron.evolve_scheme(scheme="crank-nicolson")

        # Analytical
        # t+1 since t=0 for numerical is t=1 for analytical
        neuron.t += 1.0
        V_analytical = neuron.cable_analytical()
        neuron.t -= 1.0

        # Times to plot
        plot_times = np.array([0.0, 0.5, 1, 2])
        plot_times_idx = np.array([np.argmin(np.abs(neuron.t - t)) for t in plot_times])
        
        fig, axs = plt.subplots(1, 4, figsize=(20,3))
        axs = axs.ravel()
        neuron.plot_evolution(V_analytical, ax=axs[0], plot_idxs=plot_times_idx)
        neuron.plot_evolution(V_explicit, ax=axs[1], plot_idxs=plot_times_idx)
        neuron.plot_evolution(V_implicit, ax=axs[2], plot_idxs=plot_times_idx)
        neuron.plot_evolution(V_crank, ax=axs[3], plot_idxs=plot_times_idx)

        axs[0].set_title("Analytical")
        axs[1].set_title("Implicit Euler")
        axs[2].set_title("Explicit Euler")
        axs[3].set_title("Crank-Nicolson")

        axs[0].set_xlabel(r"x [m]" + "\n" + r"$\textbf{(a)}$")
        axs[1].set_xlabel(r"x [m]" + "\n" + r"$\textbf{(b)}$")
        axs[2].set_xlabel(r"x [m]" + "\n" + r"$\textbf{(c)}$")
        axs[3].set_xlabel(r"x [m]" + "\n" + r"$\textbf{(d)}$")

        plt.tight_layout()
        plt.figure(fig)
        fig.savefig("output/task_3/t35c_schemes_compare.pdf")
        plt.show()

        # Accuracy meassure
        fig = plt.figure()
        plt.plot(neuron.t, (np.abs(V_analytical - V_explicit)).sum(axis=1)*1e3, label="Explicit")
        plt.plot(neuron.t, (np.abs(V_analytical - V_implicit)).sum(axis=1)*1e3, label="Implicit")
        plt.plot(neuron.t, (np.abs(V_analytical - V_crank)).sum(axis=1)*1e3, label="Crank-Nicolson")
        plt.grid(False)
        plt.xlabel("Time [s]")
        plt.ylabel(r"Error [mV]")
        plt.legend(loc="center")
        plt.show()
        fig.savefig("output/task_3/t35c_schemes_error.pdf")

        timer.end()

    if TASK_35_d: 
        timer.start("3.5d")

        # Parameters
        a = 0.0 # [m]
        b = 30.0 # [m]
        x0 = (b-a)/2
        Nx = 300
        Nt = 2000
        T = 10.0 # [s]
        
        # Schemes
        neuron = Neuron(a, b, x0, Nx, T, Nt)

        # Analytical
        # t+1 since t=0 for numerical is t=1 for analytical
        neuron.t += 1.0
        V_analytical = neuron.cable_analytical(threshold=1e-10)
        neuron.t -= 1.0

        timer.end()

    if TASK_35_e: 
        timer.start("3.5e")

        # Parameters
        a = 0.0 # [m]
        b = 30.0 # [m]
        x0 = (b-a)/2
        Nx = 300
        Nt = 100

        # Ts to simulate
        method_T = {
            "explicit euler": np.linspace(0.3, 1, 7),
            "implicit euler": np.linspace(100, 300, 7),
            "crank-nicolson": np.linspace(25, 32, 7)
        }
        
        # Figure
        fig, axs = plt.subplots(3, 7, figsize=(30,9))
        axs = axs.ravel()

        # For all methods
        for method_idx, method in enumerate(["explicit euler", "implicit euler", "crank-nicolson"]):

            # Different T -> different alpha
            for i, T in enumerate(method_T.get(method)):
                # Evolve and plot
                neuron = Neuron(a, b, x0, Nx, T, Nt)
                V = neuron.evolve_scheme(scheme=method)
                neuron.plot_evolution(V, ax=axs[method_idx*7 + i], plot_idxs=[len(V)-1])

                axs[method_idx*7 + i].set_title(r"$\alpha =$" + f" {neuron.lmbda**2*neuron.dt / (neuron.dx**2 * neuron.tau):.4f}")
                
                if i == 3:
                    axs[method_idx*7 + i].set_xlabel("x [m]\n" + r"$\mathbf{(" + chr(97+method_idx) + r")}$")

        plt.tight_layout()
        plt.figure(fig)
        fig.savefig("output/task_3/t35e_stability_appendix.pdf")
        plt.show()

        timer.end()


    # System parameters
    a = 0.0 # [m]
    b = 2.0e-3 # [m]
    x0 = (b-a)/2
    Nx = 300
    Nt = 1000
    T = 10.0 # [s]

    # Physiological parameters
    lmbda = 0.18e-3 # [m]
    tau = 2.0e-3 # [s]
    g_K = 5.0 # [Ohm^-1 m^-2]
    V_thr = -40e-3 # [V]
    gamma = 0.5e-3 # [V^-1]
    VN_Na = 56.0e-3 # [V]
    VN_K = -76.0e-3 # [V]
    V_appl = -50.0e-3 # [V]
    V_mem = -70.0e-3 # [V]
    Na_channel_pos = 0.0e-3 # [m]
    
    if TASK_37_a: 
        timer.start("3.7a")

        neuron = Neuron(a, b, x0, Nx, T, Nt,
                        lmbda=lmbda, tau=tau, g_K=g_K, V_thr=V_thr, gamma=gamma, 
                        VN_Na=VN_Na, VN_K=VN_K, V_appl=V_appl, V_mem=V_mem,
                        Na_channel_pos=Na_channel_pos)
        
        V_crank = neuron.evolve_scheme(scheme="crank-nicolson", extra_eq=True)

        # Plot certain times
        plot_idxs = np.linspace(0, len(V_crank)-1, 4, dtype=int)
        plot_idxs = np.zeros(5, dtype=int)
        plot_idxs[1] = np.argmin(np.abs(neuron.t - 0.5))
        plot_idxs[2] = np.argmin(np.abs(neuron.t - 1.))
        plot_idxs[3] = np.argmin(np.abs(neuron.t - 1.5))
        plot_idxs[4] = np.argmin(np.abs(neuron.t - T))
        neuron.plot_evolution(V_crank, path="output/task_3/t37a_V_drop.pdf",
                              text_pad=1, use_milli=True, plot_idxs=plot_idxs, top_text=True)
        print(f'{V_crank[-1][0] = }')

        timer.end()


    if TASK_37_b: 
        timer.start("3.7b")

        # Update some params
        b = 3.0e-3
        x0 = (b-a)/2
        T = 10.0

        fig, axs = plt.subplots(1, 4, figsize=(20,5))
        axs = axs.ravel()

        # Test multiple V_appl
        for i, V_appl in enumerate([-39e-3, -30e-3, -10e-3, 10e-3]):
            # Create class
            neuron = Neuron(a, b, x0, Nx, T, Nt,
                            lmbda=lmbda, tau=tau, g_K=g_K, V_thr=V_thr, gamma=gamma, 
                            VN_Na=VN_Na, VN_K=VN_K, V_appl=V_appl, V_mem=V_mem,
                            Na_channel_pos=Na_channel_pos)
            
            # Evolve with Crank-Nicolson
            V_crank = neuron.evolve_scheme(scheme="crank-nicolson", extra_eq=True)
            
            # Plot certain times
            plot_idxs = np.zeros(3, dtype=int)
            plot_idxs[1] = np.argmin(np.abs(neuron.t - 0.2))
            plot_idxs[2] = np.argmin(np.abs(neuron.t - T))
            neuron.plot_evolution(V_crank, ax=axs[i],  plot_idxs=plot_idxs,
                                  text_pad=0.5, use_milli=True, colors=True)

            axs[i].set_title(r"$V_{appl} = $" + f" {V_appl*1000:.0f} mV")
            axs[i].set_ylabel(r"V [mV]")
            axs[i].set_xlabel(r"x [mm]" + "\n" + r"$\textbf{(" + chr(97+i) + r")}$")

        plt.tight_layout()
        plt.figure(fig)
        fig.savefig("output/task_3/t35b_schemes_compare.pdf")
        plt.show()


        timer.end()

    if TASK_37_d: 
        timer.start("3.7d")

        # Update some params
        b = 2.0e-3 # [m]
        x0 = (b-a)/2
        T = 10.0 # [s]
        V_appl = 10.0e-3 # [V]
        Na_channel_pos = 0.25e-3 # [m]

        neuron = Neuron(a, b, x0, Nx, T, Nt,
                        lmbda=lmbda, tau=tau, g_K=g_K, V_thr=V_thr, gamma=gamma, 
                        VN_Na=VN_Na, VN_K=VN_K, V_appl=V_appl, V_mem=V_mem,
                        Na_channel_pos=Na_channel_pos)
        
        V_crank = neuron.evolve_scheme(scheme="crank-nicolson", extra_eq=True)


        # Plotting
        fig, axs = plt.subplots(1, 7, figsize=(20,3), sharey=True)
        axs = axs.ravel()
        
        # Plot certain times
        plot_idxs = np.zeros(7, dtype=int)
        plot_idxs[1] = np.argmin(np.abs(neuron.t - 0.03))
        plot_idxs[2] = np.argmin(np.abs(neuron.t - 0.1))
        plot_idxs[3] = np.argmin(np.abs(neuron.t - 0.2))
        plot_idxs[4] = np.argmin(np.abs(neuron.t - 0.3))
        plot_idxs[5] = np.argmin(np.abs(neuron.t - 0.6))
        plot_idxs[6] = np.argmin(np.abs(neuron.t - T))

        for i, idx in enumerate(plot_idxs):
            neuron.plot_evolution(V_crank, ax=axs[i],  plot_idxs=[idx], use_milli=True, text=False)

            axs[i].set_title(f"t = {neuron.t[idx]:.2f} s")
            axs[i].set_ylabel("")
            axs[i].set_xlabel(r"x [mm]")

        axs[0].set_ylabel(r"V [mV]")
        plt.tight_layout()
        plt.figure(fig)
        fig.savefig("output/task_3/t35d_Na_moved.pdf")
        plt.show()


        fig, axs = plt.subplots(4, 1, figsize=(5,20))
        axs = axs.ravel()

        # Test multiple V_appl
        for i, V_appl in enumerate([6.6e-3, 6.7e-3, 6.8e-3, 6.9e-3]):
            # Create class
            neuron = Neuron(a, b, x0, Nx, T, Nt,
                            lmbda=lmbda, tau=tau, g_K=g_K, V_thr=V_thr, gamma=gamma, 
                            VN_Na=VN_Na, VN_K=VN_K, V_appl=V_appl, V_mem=V_mem,
                            Na_channel_pos=Na_channel_pos)
            
            # Evolve with Crank-Nicolson
            V_crank = neuron.evolve_scheme(scheme="crank-nicolson", extra_eq=True)
            
            # Plot certain times
            neuron.plot_evolution(V_crank, ax=axs[i],  plot_idxs=[len(neuron.t)-1],
                                  text_pad=0.2, use_milli=True, colors=False)

            axs[i].set_title(r"$V_{appl} = $" + f" {V_appl*1000:.2f} mV")
            axs[i].set_ylabel(r"V [mV]")
            axs[i].set_xlabel(r"x [mm]" + "\n" + r"$\textbf{(" + chr(97+i) + r")}$")

        plt.tight_layout()
        plt.figure(fig)
        fig.savefig("output/task_3/t37d_Vappl_appendix.pdf")
        plt.show()

        timer.end()



if __name__ == '__main__':
    # Create dirs
    for i in range(2,4):
        if not os.path.exists(f"output/task_{i}/"):
            os.makedirs(f"output/task_{i}/")

    if TASK_2:
        Task_2()

    if TASK_3:
        Task_3()

