# Main code that run all tasks

# For VSCode development
%load_ext autoreload
%autoreload 2

# Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Code with tasks
import neuron_network as nw
from neuron_electrical import Neuron

# Choose tasks to run
TASK_2 = False
TASK_3 = True

# Subtasks (only if super is true)
TASK_22_a = False
TASK_22_b = False
TASK_22_c = False
TASK_22_d = False
TASK_22_e = False
TASK_22_f = False

TASK_35_c = True


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
# Plot params
plt.style.use('seaborn-v0_8-whitegrid')
fontsize = 12
plt.rcParams.update({
    "font.size": fontsize,
    # "axes.titlesize": fontsize,
    # "axes.labelsize": fontsize,
    # "ytick.labelsize": fontsize,
    # "xtick.labelsize": fontsize,
    # "legend.fontsize": fontsize,
    "legend.frameon": True,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral"
})
from matplotlib import rc
rc('text', usetex=True)
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats("svg")
%matplotlib inline
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# Tasks for network model
def Task_2():
    timer = Timer()

    if TASK_22_a: 
        timer.start("2.2a")

        T = nw.create_T(N_nodes=21, N_neighbours=2, plot=True, path="output/task_2/a_transformation_matrix.pdf")

        timer.end()

    if TASK_22_b: 
        timer.start("2.2b")

        # Create transformation matrix and state vector
        T = nw.create_T(N_nodes=21, N_neighbours=2)
        V = nw.create_V(N=21)

        # Evolve and plot
        V = nw.evolve_VT(V, T, N=5, plot_idx=[0,1,2,3,4], path="output/task_2/b_random_evolution.pdf")

        # Gaussian initial wave
        V_gauss = nw.create_V(N=21, type="Gaussian")
        V = nw.evolve_VT(V_gauss, T, N=50, plot_idx=[0,1,2,3,19], path="output/task_2/b_gaussian_evolution.pdf")

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

        fig.savefig("output/task_2/d_evolutions_100steps.pdf")

        # Investigate values
        print(f'{V_gauss[-1] = }')
        print(f'{V_random[-1] = }')
        print(f'{V_random2[-1] = }')
        print(f'{V_random3[-1] = }')

        eigvals, eigvecs = nw.eigvals(T, verbal=False)
    
        print(f'{eigvals = }')
        print(f'{eigvecs[:,-1] = }')


        timer.end()

    if TASK_22_e: 
        timer.start("2.2e")

        T_11 = nw.create_T(N_nodes=11, N_neighbours=2)
        T_10 = nw.create_T(N_nodes=10, N_neighbours=2)
        T = nw.join_networks(T_11, T_10, plot=True, path="output/task_2/e_T.pdf")

        nw.eigvals(T)

        # np linalg faster because of small matrix - Lanczov will scale better 

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

        # Plot
        plt.figure(fig)
        plt.tight_layout()
        plt.show()

        fig.savefig("output/task_2/f_compare_solvers.pdf")

        timer.end()

# Tasks for PDE model
def Task_3():
    timer = Timer()

    # Parameters
    a = 0.0 # [m]
    b = 20.0 # [m]
    x0 = (b-a)/2
    Nx = 300
    Nt = 400
    T = 2.0
        
    if TASK_35_c: 
        timer.start("3.5c")

        # Analytical with own values
        neuron_analytical = Neuron(a, b, x0, Nx, T, Nt)
        V_analytical = neuron_analytical.cable_analytical()
        
        # Schemes
        neuron = Neuron(a, b, x0, Nx, T, Nt)
        V_explicit = neuron.evolve_scheme(scheme="explicit euler")
        V_implicit = neuron.evolve_scheme(scheme="implicit euler")
        V_crank = neuron.evolve_scheme(scheme="crank-nicolson")
        
        fig, axs = plt.subplots(1, 4, figsize=(20,5))
        axs = axs.ravel()
        neuron_analytical.plot_evolution(V_analytical, ax=axs[0])
        neuron.plot_evolution(V_explicit, ax=axs[1], plot_times=neuron_analytical.t[1:])
        neuron.plot_evolution(V_implicit, ax=axs[2], plot_times=neuron_analytical.t[1:])
        neuron.plot_evolution(V_crank, ax=axs[3], plot_times=neuron_analytical.t[1:])

        axs[0].set_title("Analytical")
        axs[1].set_title("Implicit Euler")
        axs[2].set_title("Explicit Euler")
        axs[3].set_title("Crank-Nicolson")

        axs[0].set_xlabel(r"x [$m$]" + "\n" + r"$\textbf{(a)}$")
        axs[1].set_xlabel(r"x [$m$]" + "\n" + r"$\textbf{(b)}$")
        axs[2].set_xlabel(r"x [$m$]" + "\n" + r"$\textbf{(c)}$")
        axs[3].set_xlabel(r"x [$m$]" + "\n" + r"$\textbf{(d)}$")

        plt.tight_layout()
        plt.figure(fig)
        fig.savefig("output/task_2/t35c_schemes_compare.pdf")
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

