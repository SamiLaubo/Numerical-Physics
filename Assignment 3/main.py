# Created by Sami Laubo 15.03.2024

# For VSCode development
%load_ext autoreload
%autoreload 2

import os
import glob
from utils import Timer
from protein import Polymer
import numpy as np

# Choose tasks to run
TASK_1 = True

# Subtasks (only if super is true)
TASK_1_1 = True
TASK_1_2 = True
TASK_1_3 = True
TASK_1_4 = True
TASK_1_5 = True
TASK_1_6 = True
TASK_1_7 = True


def Task_1():
    timer = Timer()

    if TASK_1_1: 
        timer.start("1.1")

        P = Polymer(monomers=15, flexibility=0.5)
        P.find_nearest_neighbours()
        P.plot_polymer()

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

    
    if TASK_1_4:
        timer.start("1.4")
        
        P = Polymer(monomers=100, flexibility=0.0, T=1)
        P.find_nearest_neighbours()
        P.plot_polymer()
        P.MMC(MC_steps=1); P.plot_polymer(MC_step=1)
        P.MMC(MC_steps=9); P.plot_polymer(MC_step=10)
        P.MMC(MC_steps=90); P.plot_polymer(MC_step=100)
        P.plot_MMC(running_mean_N=10)

        timer.end()

    if TASK_1_5:
        timer.start("1.5")

        # Folding temperature (1000 steps)
        # T =  1: Folded
        # T =  2: Folded
        # T =  3: Folded
        # T =  4: More folded
        # T =  5: More folded
        # T =  6: More folded
        # T =  7: Slightly folded
        # T =  8: Slightly folded
        # T =  9: Unfolded
        # T = 10: Unfolded

        P = Polymer(monomers=100, flexibility=0.0, T=10)
        P.MMC_time_to_equilibrium(
            T_low=0.5, T_high=3, N=20,
            max_MC_steps=1e5, threshold=0.1, N_thr=5, N_avg=100
        )

        timer.end()

    if TASK_1_6:
        timer.start("1.6")

        # a
        P = Polymer(monomers=30, flexibility=0.0, T=1)
        P.remember_initial()
        
        # Find two teriary structures
        P.MMC(MC_steps=10000, use_threshold=False)
        P.plot_MMC()

        P.reset_to_initial()
        P.MMC(MC_steps=10000, use_threshold=False)
        P.plot_MMC()

        # b - With simulated annealing (SA)
        P.reset_to_initial()
        P.MMC(MC_steps=1000, use_threshold=False, SA=True)
        P.plot_MMC()


        timer.end()

    if TASK_1_7:
        timer.start("1.7")

        # Change some interaction signs
        for _ in range(6):
            # Choose random monomer-monomer interaction
            AA1 = np.random.randint(0, 20)
            AA2 = np.random.randint(0, 20)

            # Change both terms
            Polymer.MM_interaction_energy[AA1, AA2] *= -1

            if AA1 != AA2: # Not diagonal
                Polymer.MM_interaction_energy[AA2, AA1] *= -1

        P = Polymer(monomers=50, flexibility=0.0, T=1)
        P.plot_interaction_matrix(save=False)

        P.MMC(MC_steps=10000, use_threshold=False, SA=True)
        P.plot_MMC()
        
        P = Polymer(monomers=50, flexibility=0.0, T=1)
        P.MMC(MC_steps=10000, use_threshold=False, SA=True)
        P.plot_MMC()

        P = Polymer(monomers=50, flexibility=0.0, T=1)
        P.MMC(MC_steps=10000, use_threshold=False, SA=True)
        P.plot_MMC()

        timer.end()

if __name__ == '__main__':
    Task_1()


# Questions:
    # 1.3: Energy of system is only non-covalente energies?
    

# TODO:
    # Use sparse matrices for grid
    # Error:
        # Failed at pair [ 12 130]
        # Failed at pair [ 14 176]