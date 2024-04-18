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


# Choose tasks to run
TASK_2 = False # Diffusion
TASK_3 = True # Wave equation
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
    D = 1.
    D_type = "step"
    a, b = -2.0, 2.0
    T = 0.01
    Nx = 100
    # Nt = 1000
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
        plt.figure()
        plt.title("Reflective Boundaries")
        plt.plot(reflective_CN[0], 2*reflective_CN[1], '.', label='Crank-Nicolson')
        # plt.plot(reflective_AB[0], reflective_AB[1], label="Analytical Solution")
        plt.plot(analytical_unbounded[0], analytical_unbounded[1], label="Analytical Unbounded Solution")
        # plt.title('Crank-Nicolson Solution for Diffusion Equation')
        plt.xlabel('x')
        plt.ylabel('Concentration')
        plt.legend()
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