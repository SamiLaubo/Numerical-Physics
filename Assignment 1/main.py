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
TASK_2_5 = False
TASK_2_6 = False
TASK_2_7 = False
TASK_2_8 = False
TASK_2_9 = False
TASK_2_10_x2 = False
TASK_2_10_noncont = False
TASK_2_10_sin = False
TASK_2_10_stair = False

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
    a, b = -2.0, 2.0 # mu m
    T = 0.1 # ms
    Nx = 100
    dt = 1.818e-4

    # Create Diffusion class
    Diff = Diffusion(D, a, b, T, Nx, dt, D_type=D_type)

    if TASK_2_5: 
        timer.start("2.5")

        reflective_CN = Diff.crank_nicolson_solver(reflective=True)
        absorbing_CN = Diff.crank_nicolson_solver(reflective=False)

        timer.end()

    if TASK_2_6: 
        timer.start("2.6")
        
        if reflective_CN is None: # Else use results from 2.5
            reflective_CN = Diff.crank_nicolson_solver(reflective=True)
            absorbing_CN = Diff.crank_nicolson_solver(reflective=False)

        Diff.check_mass_conservation(reflective_CN[0], reflective_CN[1])
        Diff.check_mass_conservation(absorbing_CN[0], absorbing_CN[1])

        timer.end()

    if TASK_2_7: 
        timer.start("2.7")
        
        analytical_unbounded = Diff.analytical_unbounded()

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
        plt.ylabel(r"u [$mass/\mu m^2$]")
        plt.grid(False)
        plt.legend(loc="lower center")
        fig.savefig("output/diffusion/t27_unbounded.pdf")
        plt.show()

    if TASK_2_9:
        timer.start("2.9")

        # New params for step
        D = 1. # mu m^2 / ms
        D_pos = 0.5
        D_neg = 0.1
        D_type = "step"
        a, b = -2.0, 2.0 # mu m
        T = 0.2 # ms
        Nx = 100
        dt = 1.818e-4

        # Compare with constant diffusion
        Diff = Diffusion(D, a, b, T, Nx, dt, D_type="constant", D_pos=D, D_neg=D)
        step_unbounded = Diff.crank_nicolson_solver()
        step_unbounded_analytical = Diff.analytical_unbounded()
        fig = plt.figure()
        plt.plot(step_unbounded[0], step_unbounded[1], 'x', label="Crank-Nicolson")
        plt.plot(step_unbounded_analytical[0], step_unbounded_analytical[1], color='k', label="Exact Solution")
        plt.xlabel(r"x [$\mu m$]")
        plt.ylabel(r"u [$mass/\mu m^2$]")
        plt.grid(False)
        plt.legend(loc="lower center")
        # fig.savefig("output/diffusion/t27_unbounded.pdf")
        plt.show()

        # Step diffusion
        Diff = Diffusion(D, a, b, T, Nx, dt, D_type="step", D_pos=D_pos, D_neg=D_neg)
        step_unbounded = Diff.crank_nicolson_solver()
        step_unbounded_analytical = Diff.analytical_unbounded()
        fig, ax = plt.subplots()
        lns1 = ax.plot(step_unbounded[0], step_unbounded[1], 'x', label="Crank-Nicolson")
        lns2 = ax.plot(step_unbounded_analytical[0], step_unbounded_analytical[1], color='k', label="Exact Solution")
        ax.set_xlabel(r"x [$\mu m$]")
        ax.set_ylabel(r"u [$mass/\mu m^2$]")
        ax.grid(False)

        ax_D = ax.twinx()
        ax_D.set_ylabel(r"D [$\mu m^2/ms$]")
        lns3 = ax_D.plot(Diff.x, Diff.D_, '--', color='k', label="D(x)")
        ax_D.grid(False)

        # Legends
        lns = lns1+lns2+lns3
        ax.legend(lns, [l.get_label() for l in lns], loc="upper left")

        fig.savefig("output/diffusion/t29_step.pdf")
        plt.show()

        timer.end()

    if TASK_2_10_x2:
        timer.start("2.10.x2")

        # Continous and differentiable
        D = 1. # mu m^2 / ms
        D_pos = 0.5
        D_neg = 0.1
        D_type = "x2"
        a, b = -5.0, 5.0 # mu m
        T = 0.09 # ms
        Nx = 200
        dt = 1.818e-4

        Diff = Diffusion(D, a, b, T, Nx, dt, D_type=D_type, D_pos=D_pos, D_neg=D_neg)
        step_unbounded = Diff.crank_nicolson_solver()
        fig, ax = plt.subplots()
        lns1 = ax.plot(step_unbounded[0], step_unbounded[1], color="k", label="Crank-Nicolson")
        ax.set_xlabel(r"x [$\mu m$]")
        ax.set_ylabel(r"u [$mass/\mu m^2$]")
        # ax.grid(False)

        ax_D = ax.twinx()
        ax_D.set_ylabel(r"D [$\mu m^2/ms$]")
        lns2 = ax_D.plot(Diff.x, Diff.D_, '--', color='k', label="D(x)")
        # ax_D.grid(False)

        # Legends
        lns = lns1+lns2
        ax.legend(lns, [l.get_label() for l in lns], loc="upper left")
        ax_D.set_ylim([ax_D.get_ylim()[0], ax_D.get_ylim()[1]*6])
        ax.set_ylim([-ax.get_ylim()[1]*0.25, ax.get_ylim()[1]])
        ax.set_yticks(ax.get_yticks()[2:-1])
        ax_D.set_yticks(ax_D.get_yticks()[1:3])

        fig.savefig("output/diffusion/t210_x2.pdf")
        plt.show()

        timer.end()

    if TASK_2_10_noncont:
        timer.start("2.10.noncont")
        # New params for step
        D = 1. # mu m^2 / ms
        D_pos = 2.0
        D_neg = 0.1
        D_type = "noncont"
        a, b = -5.0, 5.0 # mu m
        T = 0.3 # ms
        Nx = 200
        dt = 1.818e-4

        # Step diffusion
        Diff = Diffusion(D, a, b, T, Nx, dt, D_type=D_type, D_pos=D_pos, D_neg=D_neg)
        step_unbounded = Diff.crank_nicolson_solver()
        fig, ax = plt.subplots()
        lns1 = ax.plot(step_unbounded[0], step_unbounded[1], color="k", label="Crank-Nicolson")
        ax.set_xlabel(r"x [$\mu m$]")
        ax.set_ylabel(r"u [$mass/\mu m^2$]")
        # ax.grid(False)

        ax_D = ax.twinx()
        ax_D.set_ylabel(r"D [$\mu m^2/ms$]")
        lns2 = ax_D.plot(Diff.x, Diff.D_, '--', color='k', label="D(x)")
        # ax_D.grid(False)

        # Legends
        lns = lns1+lns2
        ax.legend(lns, [l.get_label() for l in lns], loc="upper left")
        ax_D.set_ylim([ax_D.get_ylim()[0], ax_D.get_ylim()[1]*6])
        ax.set_ylim([-ax.get_ylim()[1]*0.25, ax.get_ylim()[1]])
        ax.set_yticks(ax.get_yticks()[2:-1])
        ax_D.set_yticks(ax_D.get_yticks()[1:3])

        fig.savefig("output/diffusion/t210_noncont.pdf")
        plt.show()

        timer.end()

    if TASK_2_10_sin:
        timer.start("2.10.sin")
        # New params for step
        D = 1.1 # mu m^2 / ms
        D_pos = -np.pi/2
        D_neg = 5.0
        D_type = "sin"
        a, b = -5.0, 5.0 # mu m
        T = 1.9 # ms
        Nx = 200
        dt = 1.818e-4

        # Step diffusion
        Diff = Diffusion(D, a, b, T, Nx, dt, D_type=D_type, D_pos=D_pos, D_neg=D_neg)
        step_unbounded = Diff.crank_nicolson_solver()
        fig, ax = plt.subplots()
        lns1 = ax.plot(step_unbounded[0], step_unbounded[1], color="k", label="Crank-Nicolson")
        ax.set_xlabel(r"x [$\mu m$]")
        ax.set_ylabel(r"u [$mass/\mu m^2$]")
        # ax.grid(False)

        ax_D = ax.twinx()
        ax_D.set_ylabel(r"D [$\mu m^2/ms$]")
        lns2 = ax_D.plot(Diff.x, Diff.D_, '--', color='k', label="D(x)")
        # ax_D.grid(False)

        # Legends
        lns = lns1+lns2
        ax.legend(lns, [l.get_label() for l in lns], loc="upper left")
        # ax_D.set_ylim([ax_D.get_ylim()[0], ax_D.get_ylim()[1]+0.5])
        ax_D.set_ylim([ax_D.get_ylim()[0], ax_D.get_ylim()[1]*6])
        ax.set_ylim([-ax.get_ylim()[1]*0.25, ax.get_ylim()[1]])
        ax.set_yticks(ax.get_yticks()[2:-1])
        ax_D.set_yticks(ax_D.get_yticks()[1:3])

        fig.savefig("output/diffusion/t210_sin.pdf")
        plt.show()

        timer.end()

    if TASK_2_10_stair:
        timer.start("2.10.stair")
        # New params for step
        D = 1.1 # mu m^2 / ms
        D_pos = 100
        D_neg = 1.0
        D_type = "stair"
        a, b = -7.0, 7.0 # mu m
        T = 0.6 # ms
        Nx = 200
        dt = 1.818e-4

        # Step diffusion
        Diff = Diffusion(D, a, b, T, Nx, dt, D_type=D_type, D_pos=D_pos, D_neg=D_neg)
        step_unbounded = Diff.crank_nicolson_solver()
        fig, ax = plt.subplots()
        lns1 = ax.plot(step_unbounded[0], step_unbounded[1], color="k", label="Crank-Nicolson")
        ax.set_xlabel(r"x [$\mu m$]")
        ax.set_ylabel(r"u [$mass/\mu m^2$]")
        # ax.grid(False)

        ax_D = ax.twinx()
        ax_D.set_ylabel(r"D [$\mu m^2/ms$]")
        lns2 = ax_D.plot(Diff.x, Diff.D_, '--', color='k', label="D(x)")
        # ax_D.grid(False)

        # Legends
        lns = lns1+lns2
        ax.legend(lns, [l.get_label() for l in lns], loc="upper left")
        ax_D.set_ylim([ax_D.get_ylim()[0], ax_D.get_ylim()[1]*6])
        ax.set_ylim([-ax.get_ylim()[1]*0.25, ax.get_ylim()[1]])
        ax.set_yticks(ax.get_yticks()[2:-1])
        ax_D.set_yticks(ax_D.get_yticks()[1:3])

        fig.savefig("output/diffusion/t210_stair.pdf")
        plt.show()

        timer.end()



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