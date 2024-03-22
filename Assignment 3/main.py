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
TASK_1_1 = False
TASK_1_2 = False
TASK_1_3 = False
TASK_1_4 = False
TASK_1_5 = True


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
        P.MMC(MC_steps=10000, plot_idxs=[1,10,100])
        P.plot_polymer(MC_step=10000)
        P.plot_MMC(running_mean_N=10)

        timer.end()

    if TASK_1_5:
        timer.start("1.4")

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
        # P = Polymer(monomers=15, flexibility=0.0, T=10)
        # P.find_nearest_neighbours()
        # P.plot_polymer()
        P = Polymer(monomers=50, flexibility=0.0, T=10)
        P.remember_initial()

        for T in np.linspace(1, 3, 10):
            # Set back to initial state
            P.reset_to_initial()
            P.T = T
            # P = Polymer(monomers=50, flexibility=0.0, T=T)

            timer.start(f"1.4 - T={T}")
            P.MMC(MC_steps=10000, verbal=False)
            timer.end()
            P.plot_MMC(running_mean_N=10)
        


if __name__ == '__main__':
    # Force Numba recompilation
    # for pycache_file in glob.glob("__pycache__/*"):
    #     os.unlink(pycache_file)

    # import IPython
    # import shutil

    # path_parent = IPython.paths.get_ipython_cache_dir()
    # path_child = os.path.join(path_parent, 'numba_cache')

    # if path_parent:
    #     if os.path.isdir(path_child):
    #         shutil.rmtree(path_child)


    Task_1()


# Questions:
    # 1.3: Energy of system is only non-covalente energies?
    

# TODO:
    # Use sparse matrices for grid
    # Error:
        # Failed at pair [ 12 130]
        # Failed at pair [ 14 176]