# Created by Sami Laubo 15.03.2024

# For VSCode development
%load_ext autoreload
%autoreload 2

import os
from utils import Timer
from protein import Polymer
import numpy as np
import matplotlib.pyplot as plt

# Choose tasks to run
TASK_1 = True
TASK_2 = False

# Subtasks (only if super is true)
TASK_1_1 = False
TASK_1_2 = False
TASK_1_3 = False
TASK_1_5 = False
TASK_1_6 = False
TASK_1_7 = False
TASK_1_8 = False
TASK_1_9 = True

TASK_2_2 = True
TASK_2_3 = True


def Task_1():
    timer = Timer()

    if TASK_1_1: 
        timer.start("1.1")

        P = Polymer(monomers=15, flexibility=1.0)
        P.find_nearest_neighbours()
        P.plot_polymer(path="output/task_1/example_plymer_N15.pdf")

        timer.end()

    if TASK_1_2:
        timer.start("1.2")

        P = Polymer()
        P.plot_interaction_matrix()
        
        timer.end()
    
    if TASK_1_3:
        timer.start("1.3")

        P = Polymer(monomers=15, flexibility=1.0)
        P.init_multiple(N=10000, plot=True, bins=100, save=True)
        
        timer.end()

    
    if TASK_1_5:
        timer.start("1.5")
        
        P = Polymer(monomers=15, flexibility=0.0, T=10)
        P.find_nearest_neighbours()
        P.plot_polymer()

        fig, axs = plt.subplots(3, 1)
        axs = axs.ravel()
        P.MMC(MC_steps=1); P.plot_polymer(MC_step=1, ax=axs[0])
        P.MMC(MC_steps=9); P.plot_polymer(MC_step=10, ax=axs[1])
        P.MMC(MC_steps=490); P.plot_polymer(MC_step=500, ax=axs[2])
        fig.savefig("output/task_1/t15_polymer_evolution.pdf")
        plt.tight_layout()
        plt.show()
        P.plot_MMC(running_mean_N=10, path="output/task_1/t15_T10_MMC.pdf", plot_polymer=False)
        # P.plot_MMC(running_mean_N=10, plot_polymer=False)

        timer.end()

    if TASK_1_6:
        timer.start("1.6")
        
        P = Polymer(monomers=15, flexibility=0.0, T=1)
        P.find_nearest_neighbours()
        P.plot_polymer()

        fig, axs = plt.subplots(3, 1)
        axs = axs.ravel()
        P.MMC(MC_steps=1); P.plot_polymer(MC_step=1, ax=axs[0])
        P.MMC(MC_steps=9); P.plot_polymer(MC_step=10, ax=axs[1])
        P.MMC(MC_steps=490); P.plot_polymer(MC_step=500, ax=axs[2])
        fig.savefig("output/task_1/t16_polymer_evolution.pdf")
        plt.tight_layout()
        plt.show()
        P.plot_MMC(running_mean_N=25, path="output/task_1/t16_T10_MMC.pdf", plot_polymer=False)
        
        P = Polymer(monomers=50, flexibility=0.0, T=1)
        P.find_nearest_neighbours()
        P.plot_polymer()
        P.MMC(MC_steps=10000)
        P.plot_MMC(running_mean_N=500, path="output/task_1/t16_T10_MMC_N50.pdf", plot_polymer=False)

        timer.end()

    if TASK_1_7:
        timer.start("1.7")

        fig, ax = plt.subplots()
        plt.title("MC Steps before equilibration")
        plt.xlabel(r"Temperature $(k_b)$")
        plt.ylabel("Steps")

        for N in [15, 25, 35, 45, 55]:
            P = Polymer(monomers=N, flexibility=0.5, T=10)
            # P.find_nearest_neighbours()
            # P.plot_polymer()
            P.MMC_time_to_equilibrium(
                T_low=0.5, T_high=3, N=10,
                max_MC_steps=1e4, threshold=0.2, N_thr=3, N_avg=100,
                phase_ax = ax,
                path="output/task_1/"
            )
        plt.figure(fig)
        plt.legend()
        fig.savefig("output/task_1/t17_phase_diagram.pdf")
        plt.show()

        timer.end()

    if TASK_1_8:
        timer.start("1.8")

        # a
        P = Polymer(monomers=30, flexibility=0.2, T=1)
        P.remember_initial()
        
        # Find two teriary structures
        P.MMC(MC_steps=10000, use_threshold=False)
        P.plot_MMC(running_mean_N=500, path="output/task_1/t18_tertiary_1.pdf")

        P.reset_to_initial()
        P.MMC(MC_steps=10000, use_threshold=False)
        P.plot_MMC(running_mean_N=500, path="output/task_1/t18_tertiary_2.pdf")

        # b - With simulated annealing (SA)
        P.reset_to_initial()
        P.MMC(MC_steps=2000, use_threshold=False, SA=True)
        P.plot_MMC(running_mean_N=100, path="output/task_1/t18_tertiary_SA.pdf")


        timer.end()

    if TASK_1_9:
        timer.start("1.9")

        # Change some interaction signs
        # for _ in range(6):
        #     # Choose random monomer-monomer interaction
        #     AA1 = np.random.randint(0, 20)
        #     AA2 = np.random.randint(0, 20)

        #     # Change both terms
        #     Polymer.MM_interaction_energy[AA1, AA2] *= -1

        #     if AA1 != AA2: # Not diagonal
        #         Polymer.MM_interaction_energy[AA2, AA1] *= -1

        P = Polymer(monomers=50, flexibility=0.2, T=1)
        P.plot_interaction_matrix(save=True, path="task_1/t19_MM_interactions.pdf")

        P.MMC(MC_steps=10000, use_threshold=False, SA=True)
        print(P.monomer_AA_number)
        P.plot_MMC(running_mean_N=500)
        
        P = Polymer(monomers=50, flexibility=0.3, T=1)
        print(P.monomer_AA_number)
        P.MMC(MC_steps=10000, use_threshold=False, SA=True)
        P.plot_MMC(running_mean_N=500)

        P = Polymer(monomers=50, flexibility=0.4, T=1)
        print(P.monomer_AA_number)
        P.MMC(MC_steps=10000, use_threshold=False, SA=True)
        P.plot_MMC(running_mean_N=500)

        timer.end()


def Task_2():
    timer = Timer()

    if TASK_2_2: 
        timer.start("2.2")

        P = Polymer(monomers=15, flexibility=0.0, dims=2, T=10)
        P.find_nearest_neighbours()
        P.plot_polymer()
        P.MMC(MC_steps=1); P.plot_polymer(MC_step=1)
        P.MMC(MC_steps=9); P.plot_polymer(MC_step=10)
        P.MMC(MC_steps=90); P.plot_polymer(MC_step=100)
        P.plot_MMC(running_mean_N=10)

        timer.end()

    if TASK_2_3: 
        timer.start("2.3")

        P = Polymer(monomers=15, flexibility=0.0, T=10, dims=3)
        P.MMC_time_to_equilibrium(
            T_low=0.5, T_high=10, N=10,
            max_MC_steps=1e5, threshold=0.1, N_thr=5, N_avg=100
        )

        timer.end()

if __name__ == '__main__':
    # Create dirs
    for i in range(1,3):
        if not os.path.exists(f"output/task_{i}/"):
            os.makedirs(f"output/task_{i}/")

    if TASK_1:
        Task_1()

    if TASK_2:
        Task_2()    

# TODO:
    # Use sparse matrices for grid
    # Fix 3d plots. Tightlayout first?
    # Run 1.6 again with more averaging
    # Error:
        # Failed at pair [ 12 130]
        # Failed at pair [ 14 176]