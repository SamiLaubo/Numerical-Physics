# Created by Sami Laubo 15.03.2024

# For VSCode development
%load_ext autoreload
%autoreload 2

import os
import glob
from utils import Timer
from protein import Polymer

# Choose tasks to run
TASK_1 = True

# Subtasks (only if super is true)
TASK_1_1 = False
TASK_1_2 = False
TASK_1_3 = False
TASK_1_4 = True


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
        
        P = Polymer(monomers=15, flexibility=0.5)
        P.find_nearest_neighbours()
        P.plot_polymer()
        P.MMC(MC_steps=1)

        timer.end()


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