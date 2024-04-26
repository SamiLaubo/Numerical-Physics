# Created by Sami Laubo 07.02.2024

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
import time
import matplotlib.animation as animation

class Wave_Solver:
    def __init__(self, a, b, Nx, Nt, T, c=1) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.T = T
        self.Nx = Nx
        self.Nt = Nt
        self.t, self.dt = np.linspace(0, T, Nt, retstep=True)
        xy_values, self.h = np.linspace(0, 1, self.Nx, retstep=True)
        self.XX, self.YY = np.meshgrid(xy_values, xy_values)
        self.beta = self.c**2 * self.dt**2 / self.h**2

        print(f'{self.h / self.dt / np.sqrt(2) = }')


    def explicit_solver(self, init_cond="normal"):
        # (t, x, y)
        u = np.zeros((self.Nt, self.Nx, self.Nx))

        # Initial condition
        if init_cond == "normal":
            sin_x = np.sin(np.pi*self.XX)
            sin_y = np.sin(2*np.pi*self.YY)
            u[0] = sin_x * sin_y

        else:
            u[0] = np.exp(-((self.XX - 0.5)**2 + (self.YY - 0.5)**2)/0.001)

        # Boundary condition
        u[0, :, 0] = 0
        u[0, :, -1] = 0
        u[0, 0, :] = 0
        u[0, -1, :] = 0

        @njit
        def solve_u(u, beta):
            # n = 0
            for i in range(1, u.shape[1] - 1):
                for j in range(1, u.shape[2] - 1):
                    u[1,i,j] = -u[0,i,j] + (1-2*beta)*2*u[0,i,j] + beta*(u[0,i+1,j] + u[0,i-1,j] + u[0,i,j+1] + u[0,i,j-1])

            # n > 0
            for n in range(1, u.shape[0] - 1):
                for i in range(1, u.shape[1] - 1):
                    for j in range(1, u.shape[2] - 1):
                        u[n+1,i,j] = -u[n-1,i,j] + (1-2*beta)*2*u[n,i,j] + beta*(u[n,i+1,j] + u[n,i-1,j] + u[n,i,j+1] + u[n,i,j-1])

            return u
        
        return solve_u(u, self.beta)

    def analytical_solution(self):

        sin_x = np.sin(np.pi*self.XX)
        sin_y = np.sin(2*np.pi*self.YY)

        spatial_part = sin_x * sin_y
        time_part = np.cos(np.sqrt(5) * np.pi * self.t)

        # (t, x, y)
        u = spatial_part * time_part[:,None,None]

        return u

    def animate(self, u, path=""):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
            
        surface = [ax.plot_surface(self.XX, self.YY, u[0], cmap="seismic")]
        ax.set_zlim(-1, 1)
        ax.set_aspect('equal')
        
        def anim_func(i, u, surface):
            surface[0].remove()
            surface[0] = ax.plot_surface(self.XX, self.YY, u[i], cmap="seismic")
        
        anim = animation.FuncAnimation(
            fig,
            anim_func,
            u.shape[0],
            interval = 1,
            repeat=False,
            blit=False,
            fargs=(u, surface)
        )

        if len(path) > 0:
            if os.path.exists(path):
                os.remove(path)
            anim.save(path, fps=30)
            plt.close()

    def plot_evolution(self, u, u_anal=None, path=""):

        fig = plt.figure(figsize=(10,5))

        rows = 2 if u_anal is not None else 1

        for i in range(5):
            ax = fig.add_subplot(rows,5,i+1, projection="3d")
            ax.plot_surface(self.XX, self.YY, u[len(u)//5*i], cmap="seismic")
            ax.set_zlim(-1, 1)
            ax.set_aspect("equal")
            ax.set_title(f"t = {self.t[len(u)//5*i]:.2f}s")
            ax.grid(False)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            if i == 2 and u_anal is not None:
                ax.text2D(0.5, 1.2, "(a)", transform=ax.transAxes, size=12)

            if u_anal is not None:
                ax = fig.add_subplot(rows,5,i+6, projection="3d")
                ax.plot_surface(self.XX, self.YY, u_anal[len(u_anal)//5*i], cmap="seismic")
                ax.set_zlim(-1, 1)
                ax.set_aspect("equal")
                ax.set_title(f"t = {self.t[len(u)//5*i]:.2f}s")
                ax.grid(False)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                if i == 2: 
                    ax.text2D(0.5, 1.2, "(b)", transform=ax.transAxes, size=12)


        # plt.tight_layout()
        plt.show()
        if len(path) > 0:
            fig.savefig(path)


if __name__ == '__main__':

    WS = Wave_Solver(a=0, b=1, Nx=100, Nt=200, T=2/np.sqrt(5))

    # u = WS.analytical_solution()
    t1 = time.time()
    u = WS.explicit_solver(init_cond="wave")
    t2 = time.time()
    print(f'Solving time: {t2 - t1}')

    t1 = time.time()
    WS.animate(u)
    t2 = time.time()
    print(f'Animation time: {t2 - t1}')