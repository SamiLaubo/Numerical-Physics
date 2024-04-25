# For VSCode development
%load_ext autoreload
%autoreload 2

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from Wave_equation import Wave_Solver
from Diffusion import Diffusion
from Hopf import Advection

# Plot params
plt.style.use('seaborn-v0_8-whitegrid')
fontsize = 12
plt.rcParams.update({
    "axes.titlesize": fontsize,
    "axes.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "xtick.labelsize": fontsize,
    "legend.fontsize": fontsize,
    "legend.frameon": True,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral"
})
from IPython.display import set_matplotlib_formats
set_matplotlib_formats("svg")

# Choose tasks to run
TASK_2 = True # Diffusion
TASK_3 = False # Wave equation
TASK_4 = False # Hopf

# Subtasks (only if super is true)
TASK_2_5 = True
TASK_2_6 = True
TASK_2_7 = True
TASK_2_8 = True

TASK_3_2 = True
TASK_3_4 = True

TASK_4_2 = True
TASK_4_5 = True


# Timing function
class Timer:
    def start(self, task="0.0"):
        self.task = task
        self.t1 = time.time()
        print(f"\nTask {self.task}")

    def end(self):
        self.t2 = time.time()
        print(f'Task {self.task} time: {self.t2 - self.t1:.4e}s')
timer = Timer()


def diff_main():
    D = 1. # mu m^2 / ms
    D_type = "constant"
    a, b = 0.0, 4.0 # mu m
    T = 0.1 # ms
    Nx = 100
    dt = 1.818e-4

    # Create Diffusion class
    Diff = Diffusion(D, a, b, T, Nx, dt)

    if TASK_2_5: 
        timer.start("2.5")

        reflective_CN = Diff.crank_nicolson_solver(D_type=D_type, reflective=True)
        absorbing_CN = Diff.crank_nicolson_solver(D_type=D_type, reflective=False)

        timer.end()

    if TASK_2_6: 
        timer.start("2.6")
        
        if reflective_CN is None: # Else use results from 2.5
            reflective_CN = Diff.crank_nicolson_solver(D_type=D_type, reflective=True)
            absorbing_CN = Diff.crank_nicolson_solver(D_type=D_type, reflective=False)

        Diff.check_mass_conservation(reflective_CN[0], reflective_CN[1])
        Diff.check_mass_conservation(absorbing_CN[0], absorbing_CN[1])

        timer.end()

    if TASK_2_7: 
        timer.start("2.7")
        
        analytical_unbounded = Diff.analytical_unbounded(D_type=D_type)

        timer.end()

    if TASK_2_8: 
        timer.start("2.8")
        
        reflective_AB = Diff.analytical_bounded(reflective=True)
        absorbing_AB = Diff.analytical_bounded(reflective=False)

        timer.end()

    if TASK_2_5 and TASK_2_8 and TASK_2_7:
        # fig, axs = plt.subplots(1,3, figsize=(20,5))

        # # Reflective boundaries
        # axs[0].set_title("Reflective Boundaries")
        # axs[0].plot(reflective_CN[0], reflective_CN[1], 'x', label='Crank-Nicolson')
        # axs[0].plot(reflective_AB[0], reflective_AB[1], color='k', label="Exact Solution")
        
        # # Absorbing boundaries
        # axs[1].set_title("Absorbing Boundaries")
        # axs[1].plot(absorbing_CN[0], absorbing_CN[1], 'x', label='Crank-Nicolson')
        # axs[1].plot(absorbing_AB[0], absorbing_AB[1], color='k', label="Exact Solution")

        # # Unbounded problem
        # axs[2].set_title("Unbounded")
        # axs[2].plot(reflective_CN[0], reflective_CN[1], 'x', label="Crank-Nicolson")
        # axs[2].plot(analytical_unbounded[0], analytical_unbounded[1], color='k', label="Exact Solution")

        # for i, ax in enumerate(axs.ravel()):
        #     ax.set_xlabel(f'x\n({chr(97+i)})')
        #     ax.set_ylabel('Concentration')
        #     ax.legend(loc="lower center")
        #     ax.grid(False)
        
        # plt.show()
        # fig.savefig(f"output/diffusion/t28_T{T}.pdf")

        fig = plt.figure()
        plt.plot(reflective_CN[0], reflective_CN[1], 'x', label="Crank-Nicolson")
        plt.plot(analytical_unbounded[0], analytical_unbounded[1], color='k', label="Exact Solution")
        plt.xlabel(r"x [$\mu m$]")
        plt.ylabel(r"u [mass/\mu m^2]")
        plt.grid(False)
        plt.legend(loc="lower center")
        fig.savefig("output/diffusion/t27_unbounded.pdf")
        plt.show()



def wave_main():
    if TASK_3_2: 
        timer.start("3.2")

        WS = Wave_Solver(a=0, b=1, Nx=100, Nt=200, T=2/np.sqrt(5))

        t1 = time.time()
        u = WS.explicit_solver(init_cond="normal")
        # u = WS.analytical_solution()
        t2 = time.time()
        print(f'Solving time: {t2 - t1}')

        t1 = time.time()
        WS.animate(u)
        t2 = time.time()
        print(f'Animation time: {t2 - t1}')

        timer.end()

    if TASK_3_4: 
        timer.start("3.4")

        WS = Wave_Solver(a=0, b=1, Nx=100, Nt=200, T=2/np.sqrt(5))

        t1 = time.time()
        u = WS.explicit_solver(init_cond="wave")
        t2 = time.time()
        print(f'Solving time: {t2 - t1}')

        t1 = time.time()
        WS.animate(u, path="output/wave/explicit_solution_wave.gif")
        t2 = time.time()
        print(f'Animation time: {t2 - t1}')

        timer.end()

def hopf_main():
    if TASK_4_2: 
        timer.start("4.2")

        A = Advection(a=0, b=1, c=0.01, Nx=500, Nt=1000, T=2, A=0.3, init_name="gaussian")

        t1 = time.time()
        # u_anal = A.analytical()
        # u_lex = A.Lex_Wendroff()
        u = A.Hopf_Lax_Wendroff(x0=0.5)
        t2 = time.time()
        print(f'Solving time: {t2 - t1}')

        t1 = time.time()
        # A.animate(u_anal, u_lex)
        A.animate(u, path = "output/hopf/hopf_gaussian.gif")
        t2 = time.time()
        print(f'Animation time: {t2 - t1}')

        timer.end()

    if TASK_4_5: 
        timer.start("4.5")

        A = Advection(a=0, b=1, c=0.01, Nx=500, Nt=1000, T=2, A=0.3, init_name="gaussian")

        t1 = time.time()
        # u_anal = A.analytical()
        # u_lex = A.Lex_Wendroff()
        u = A.Hopf_Lax_Wendroff(x0=0.5)
        t2 = time.time()
        print(f'Solving time: {t2 - t1}')

        t1 = time.time()
        # A.animate(u_anal, u_lex)
        A.animate(u, path = "output/hopf/hopf_gaussian.gif")
        t2 = time.time()
        print(f'Animation time: {t2 - t1}')

        timer.end()


if __name__ == '__main__':
    # Create dirs
    for i in ["diffusion", "wave", "hopf"]:
        if not os.path.exists(f"output/{i}/"):
            os.makedirs(f"output/{i}/")

    if TASK_2:
        diff_main()

    if TASK_3:
        wave_main()

    if TASK_4:
        hopf_main()