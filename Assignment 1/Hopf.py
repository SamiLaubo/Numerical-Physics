
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import time


class Advection:
    def __init__(self, a, b, c, Nx, Nt, T, A=1):
        self.a = a
        self.b = b
        self.c = c
        self.Nx = Nx
        self.Nt = Nt
        self.T = T
        self.A = A

        self.x, self.dx = np.linspace(a, b, Nx, retstep=True)
        self.t, self.dt = np.linspace(0, T, Nt, retstep=True)

    def u0(self, x, name="gaussian"):

        if name == "gaussian":
            return self.A*np.exp(-x**2/0.005)

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
        
    
    def animate(self, u):
        fig, ax = plt.subplots()
        plt.ylim([-0.1, self.A*2])

        line, = ax.plot(self.x, u[0])
        
        def anim_func(i):
            line.set_ydata(u[i])
            return line,
        
        anim = animation.FuncAnimation(
            fig,
            anim_func,
            u.shape[0],
            interval = 1,
            repeat=False,
            blit=False
        )

        path = "output/hopf/advection_lex_wave.gif"
        if os.path.exists(path):
            os.remove(path)
        anim.save(path, fps=60)
        plt.close()

if __name__ == '__main__':

    A = Advection(a=-0.2, b=2, c=0.01, Nx=500, Nt=500, T=200)

    t1 = time.time()
    u = A.analytical()
    # u = A.Lex_Wendroff()
    t2 = time.time()
    print(f'Solving time: {t2 - t1}')

    t1 = time.time()
    A.animate(u)
    t2 = time.time()
    print(f'Animation time: {t2 - t1}')
