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
TASK_2 = True

# Subtasks (only if super is true)
TASK_22_a = False
TASK_22_b = False
TASK_22_c = False
TASK_22_d = True
TASK_22_e = True


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
        fig, axs = plt.subplots(2, 2, sharex=True)
        axs = axs.ravel()

        # Transformation matrix
        T = nw.create_T(N_nodes=21, N_neighbours=2)

        # Gaussian initial wave
        V_gauss = nw.create_V(N=21, type="Gaussian")
        V = nw.evolve_VT(V_gauss, T, N=100, plot_idx=[99], axs=[axs[0], axs[2]], titles=["Gaussian", "(a)"])#, path="output/task_2/d_gaussian_evolution.pdf")

        # Random initial wave
        V_gauss = nw.create_V(N=21, type="random")
        V = nw.evolve_VT(V_gauss, T, N=100, plot_idx=[99], axs=[axs[1], axs[3]], titles=["Random", "(b)"])#, path="output/task_2/d_random_evolution.pdf")
        
        # Plot
        plt.figure(fig)
        plt.tight_layout()
        plt.show()


        timer.end()

if __name__ == '__main__':
    # Create dirs
    for i in range(2,4):
        if not os.path.exists(f"output/task_{i}/"):
            os.makedirs(f"output/task_{i}/")

    if TASK_2:
        Task_2()