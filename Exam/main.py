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

# Choose tasks to run
TASK_2 = False
TASK_3 = True

# Subtasks (only if super is true)
TASK_22_a = True
TASK_22_b = True
TASK_22_c = True
TASK_22_d = True
TASK_22_e = True
TASK_22_f = True


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

        # Create transformation matrix
        T_11 = nw.create_T(N_nodes=11, N_neighbours=2)
        T_10 = nw.create_T(N_nodes=10, N_neighbours=2)
        T = nw.join_networks(T_11, T_10)

        # Create initial state
        V = nw.create_V(21, type="inv_step", normalize=True)
        
        # Evolve
        V = nw.evolve_VT(V, T, N=5, plot_idx=[0,4], path="output/task_2/f_evolution.pdf")

        timer.end()

# Tasks for PDE model
def Task_3():
    timer = Timer()

    if TASK_22_a: 
        timer.start("2.2a")

if __name__ == '__main__':
    # Create dirs
    for i in range(2,4):
        if not os.path.exists(f"output/task_{i}/"):
            os.makedirs(f"output/task_{i}/")

    if TASK_2:
        Task_2()