# Created by Sami Laubo 07.02.2024

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from numba import njit
import scipy.special
import os

import matplotlib.animation as animation

class Wave_Solver:
    def __init__(self, a, b, Nx, Nt, T, c=1) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.T = T
        self.Nx = Nx
        self.Nt = Nt
        self.t = np.linspace(0, T, Nt)
        xy_values = np.linspace(0, 1, self.Nx)
        self.XX, self.YY = np.meshgrid(xy_values, xy_values)


    def analytical_solution(self):

        sin_x = np.sin(np.pi*self.XX)
        sin_y = np.sin(2*np.pi*self.YY)

        spatial_part = sin_x * sin_y
        time_part = np.cos(np.sqrt(5) * np.pi * self.t)

        # (t, x, y)
        u = spatial_part * time_part[:,None,None]

        return u
    

    def animate(self, u):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
            
        # surface = ax.plot_surface(self.XX, self.YY, u[2])
        # ax.set_zlim(-1, 1)
        # ax.set_aspect('equal')
        # plt.show()
        # plt.show()

        def anim_func(i):
            # print(f'{i = }')
            # im.set_array(u[i])
            ax.cla()
            ax.plot_surface(self.XX, self.YY, u[i])
            ax.set_zlim(-1, 1)
            ax.set_aspect('equal')
            # surface.remove()

            return fig,
        
        anim = animation.FuncAnimation(
            fig,
            anim_func,
            10, # u.shape[0],
            # interval = 1000 / 2,
            interval = 1,
            repeat=False,
            blit=False
        )

        # return anim
        # plt.show()
        if os.path.exists("output/wave_equation/analytical_solution.mp4"):
            os.remove("output/wave_equation/analytical_solution.mp4")

        anim.save("output/wave_equation/analytical_solution.mp4", fps=3)
        plt.close()


if __name__ == '__main__':

    WS = Wave_Solver(a=0, b=1, Nx=100, Nt=10, T=10)

    u = WS.analytical_solution()
    WS.animate(u)