
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import time
from numba import njit


class Advection:
    def __init__(self, a, b, c, Nx, Nt, T, A=1, init_name="gaussian"):
        self.a = a
        self.b = b
        self.c = c
        self.Nx = Nx
        self.Nt = Nt
        self.T = T
        self.A = A
        self.init_name = init_name

        self.x, self.dx = np.linspace(a, b, Nx, retstep=True)
        self.t, self.dt = np.linspace(0, T, Nt, retstep=True)

    def u0(self, x, x0=0):

        if self.init_name == "gaussian":
            return self.A*np.exp(-(x-x0)**2/0.005)
        elif self.init_name == "step":
            u0 = np.ones_like(x)
            u0[:len(u0)//2] = 1
            u0[len(u0)//2:] = 0
            return u0

    def analytical(self):

        u = np.zeros((self.Nt, self.Nx))

        for idx, t in enumerate(self.t):
            u[idx] = self.u0(self.x - self.c*t)

        return u
    
    def Lex_Wendroff(self):

        gamma = self.c * self.dt / self.dx

        print(f'{gamma = }')

        A = np.zeros((self.Nx, self.Nx))
        np.fill_diagonal(A, 1 - gamma**2)
        np.fill_diagonal(A[1:], gamma/2*(gamma+1)) # Lower
        np.fill_diagonal(A[:, 1:], gamma/2*(gamma-1)) # Upper

        u = np.zeros((self.Nt, self.Nx))
        u[0] = self.u0(self.x)

        for i in range(1, self.Nt):
            u[i] = A @ u[i-1]

        return u
    
    def Hopf_Lax_Wendroff(self, x0=0.5):

        print(f'{self.dt / self.dx / 4 = }')

        u = np.zeros((self.Nt, self.Nx))
        u[0] = self.u0(self.x, x0=x0)

        @njit
        def loop(u, dt, dx):
            for n in range(0, u.shape[0] - 1):
                for i in range(1, u.shape[1] - 1):
                    u[n+1,i] = u[n,i] - dt/(2*dx)*(u[n,i+1]**2-u[n,i-1]**2) \
                               + dt**2/(2*dx**2)*((u[n,i+1]+u[n,i])*(u[n,i+1]**2-u[n,i]**2) - \
                                                  (u[n,i]+u[n,i-1])*(u[n,i]**2-u[n,i-1]**2))
                    
            return u
        
        return loop(u, self.dt, self.dx)



        
    
    def animate(self, u):#, u_anal, u_lex):
        fig, ax = plt.subplots()
        plt.ylim([-0.1, np.max(u)*1.1])

        line, = ax.plot(self.x, u[0])
        # line_anal, = ax.plot(self.x, u_anal[0], label="Analytical")
        # line_lex, = ax.plot(self.x, u_lex[0], '.', label="Lax-Wendroff")
        
        def anim_func(i):

            line.set_ydata(u[i])
            # line_anal.set_ydata(u_anal[i])
            # line_lex.set_ydata(u_lex[i])
            # return line_anal, line_lex
            return line,
        
        anim = animation.FuncAnimation(
            fig,
            anim_func,
            # u_anal.shape[0],
            u.shape[0],
            interval = 1,
            repeat=False,
            blit=False
        )

        # plt.legend()

        path = "output/hopf/hopf_gaussian.gif"
        if os.path.exists(path):
            os.remove(path)
        anim.save(path, fps=60)
        plt.close()

if __name__ == '__main__':

    A = Advection(a=0, b=1, c=0.01, Nx=500, Nt=1000, T=2, A=0.3, init_name="gaussian")

    t1 = time.time()
    # u_anal = A.analytical()
    # u_lex = A.Lex_Wendroff()
    u = A.Hopf_Lax_Wendroff(x0=0.5)
    t2 = time.time()
    print(f'Solving time: {t2 - t1}')

    t1 = time.time()
    # A.animate(u_anal, u_lex)
    A.animate(u)
    t2 = time.time()
    print(f'Animation time: {t2 - t1}')
