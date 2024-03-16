# Created by Sami Laubo 15.03.2024

# For VSCode development
%load_ext autoreload
%autoreload 2

import time

from protein import Polymer

# Choose tasks to run
TASK_1 = True

# Subtasks (only if super is true)
TASK_1_1 = True


def Task_1():
    
    if TASK_1_1:
        P = Polymer(monomers=15, flexibility=0.5)
        # P.plot_polymer()


if __name__ == '__main__':
    Task_1()