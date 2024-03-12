# Created by Sami Laubo 07.03.2024

# For VSCode development
%load_ext autoreload
%autoreload 2

import time
from schrodinger import *
import root_finder

# In this code: t' = 2mL^2/hbar * t

# Choose tasks to run
TASK_2 = False
TASK_3 = False
TASK_4 = True

# Subtasks (only if super is true)
TASK_2_4 = True
TASK_2_5 = True
TASK_2_7 = True
TASK_2_10 = True
TASK_2_11 = True

TASK_3_1 = False
TASK_3_2 = False
TASK_3_3 = False
TASK_3_4 = False
TASK_3_5 = False
TASK_3_6 = False
TASK_3_7 = False
TASK_3_9 = True

TASK_4_1 = True
TASK_4_2 = True
TASK_4_4 = False

# Constants
NX = 1002 # Number of x-points


# Function to run task 2
def Task_2():
    # Create class
    S = Schrodinger(L=1, Nx=NX, Nt=100, T=2e-4)
    S.eigen()

    if TASK_2_4:
        t1 = time.time(); print("\nTask 2.4")

        # Find eigenvalues and plot
        S.eigen()
        S.plot_eig_values(plot_vals_n=True, path="output/task_2/t24_f")
        
        t2 = time.time(); print(f'Task 2.4 time: {t2 - t1:.4e}')


    if TASK_2_5:
        t1 = time.time(); print("\nTask 2.5")

        # Plot how the error in eigenvectors increase with dx
        S.eigvec_error_dx(Nx_low=50, Nx_high=1000, N=100, save=False, num_eigvecs=20)
        
        t2 = time.time(); print(f'\nTask 2.5 time: {t2 - t1:.4e}')


    if TASK_2_7:
        t1 = time.time(); print("\nTask 2.7")
        
        # Check orthogonality of 20 lowest eigenvectors
        S.check_orthogonality(threshold=1e-9)
        
        t2 = time.time(); print(f'Task 2.7 time: {t2 - t1:.4e}')


    if TASK_2_10:
        t1 = time.time(); print("\nTask 2.10")
        
        # Set initial condition to first eigen function
        S.init_cond(name="psi_0")
        S.evolve(path="output/task_2/t210_prob_dens_psi1.pdf")
        
        t2 = time.time(); print(f'Task 2.10 time: {t2 - t1:.4e}')

    if TASK_2_11:
        t1 = time.time(); print("\nTask 2.10")
        
        # Set initial condition to first eigen function
        S.init_cond(name="delta")
        S.evolve(path="output/task_2/t210_prob_dens_delta.pdf")

        # Scaled better
        S.evolve(start_idx_plot=1, path="output/task_2/t210_prob_dens_delta_scaled.pdf")
        
        t2 = time.time(); print(f'Task 2.10 time: {t2 - t1:.4e}')


def Task_3():
    v0 = 1e3

    if TASK_3_1:
        t1 = time.time(); print("\nTask 3.1")

        # Check if barrier potential gives well when v0=0
        S_well = Schrodinger(L=1, Nx=NX, pot_type="well")
        S_barrier = Schrodinger(L=1, Nx=NX, pot_type="barrier", v0=0)

        # Get eigenvalues
        S_well.eigen()
        S_barrier.eigen()

        # Plot both
        S_well.plot_eig_values(S_barrier, path="output/task_3/t31_")

        t2 = time.time(); print(f'\nTask 3.1 time: {t2 - t1:.4e}')

    
    if TASK_3_2:
        t1 = time.time(); print("\nTask 3.2")

        # With barrier
        v0 = 1e3 # Barrier height
        S = Schrodinger(L=1, Nx=NX, pot_type="barrier", v0=v0, Nt=100)
        S.eigen()
        S.plot_eig_values(n_eig_vecs=4, path="output/task_3/t31_")

        t2 = time.time(); print(f'Task 3.2 time: {t2 - t1:.4e}')

    
    if TASK_3_3:
        t1 = time.time(); print("\nTask 3.3")

        v0 = 1e3
        S = Schrodinger(L=1, Nx=NX, pot_type="barrier", v0=v0, Nt=100)
        S.eigen()
        S.init_cond(name="eigenfuncs", eigenfunc_idxs=[0,1])
        S.plot_Psi_0()

        # Update end time
        S.T = np.pi / (S.eig_vals[1] - S.eig_vals[0])

        # Discretize t again
        S.discretize_x_t()

        # Evolve    
        S.evolve(plot=True, animate=False, path="output/task_3/t33_prob_dens.pdf")
        
        t2 = time.time(); print(f'Task 3.3 time: {t2 - t1:.4e}')

    
    if TASK_3_4:
        t1 = time.time(); print("\nTask 3.4")

        v0 = 1e3
        S = Schrodinger(L=1, Nx=NX, pot_type="barrier", v0=v0, Nt=100)
        S.eigen()

        # Root finding
        # Plot function
        root_finder.plot_f(v0, eig_vals=S.eig_vals[:np.where(S.eig_vals>v0)[0][0]], path="output/task_3/t34_roots.pdf")
        
        t2 = time.time(); print(f'Task 3.4 time: {t2 - t1:.4e}')

    
    if TASK_3_5:
        t1 = time.time(); print("\nTask 3.5")
    
        # Find root values
        root_finder.find_roots(v0, eig_vals=S.eig_vals[:np.where(S.eig_vals>v0)[0][0]])
    
        t2 = time.time(); print(f'Task 3.5 time: {t2 - t1:.4e}')

    
    if TASK_3_6:
        t1 = time.time(); print("\nTask 3.6")
    
        S = Schrodinger(pot_type="barrier")
        S.eigvals_under_barrier(v0_low=0, v0_high=1e5, N=10, path="output/task_3/t36_eigvals_under_barrier.pdf")
    

        # Find value where #lambda 0->1
        v0_low, v0_high = 22, 23
        for _ in range(10):
            print(f'{v0_low, v0_high = }')
            v0_low, v0_high = S.eigvals_under_barrier(v0_low=v0_low, v0_high=v0_high, N=10, plot=False)
        
        t2 = time.time(); print(f'Task 3.6 time: {t2 - t1:.4e}')
    

    
    if TASK_3_7:
        t1 = time.time(); print("\nTask 3.7")

        v0 = 1e3
        S = Schrodinger(Nx=NX, pot_type="barrier", v0=v0, Nt=10)
        S.eigen()
        S.init_cond(name="eigenfuncs", eigenfunc_idxs=[1])
        S.plot_Psi_0()
        # Update end time
        # S.T = np.pi / (S.eig_vals[1]) * S.t0

        # Fiddle with 100 to see when it messes up
        S.T = 1 * S.dx_**2
        # Discretize t again
        S.discretize_x_t()
        print(f'{S.dt_/S.dx_**2 = }')
        
        # Use forward euler to solve
        S.forward_scheme(method="Forward Euler", plot=True, animate=False, path="output/task_3/t37_forward_euler.pdf", CFL=True)
        
        t2 = time.time(); print(f'Task 3.7 time: {t2 - t1:.4e}')

    
    if TASK_3_9:
        t1 = time.time(); print("\nTask 3.9")

        # v0 = 1e3
        # S = Schrodinger(Nx=NX, pot_type="barrier", v0=v0, Nt=10)
        # S.eigen()
        # S.init_cond(name="eigenfuncs", eigenfunc_idxs=[1])
        # # S.init_cond(name="delta", eigenfunc_idxs=[1])
        # # S.plot_Psi_0()
        # # Update end time
        # # S.T = np.pi / (S.eig_vals[1]) * S.t0

        # # Fiddle with 100 to see when it messes up
        # S.T = 1 * S.dx_**2
        # # Discretize t again
        # S.discretize_x_t()
        # print(f'{S.dt_/S.dx_**2 = }')

        # # Use Crank Nicolson to solve
        # S.forward_scheme(method="Crank Nicolson", plot=True, animate=False)
        
        # With delta inital compared to 2.11
        S = Schrodinger(Nx=NX, pot_type="barrier", v0=0, Nt=40, T=2e-7)
        S.eigen()
        S.init_cond(name="delta")
        # Discretize t again
        S.discretize_x_t()
        print(f'{S.dt_/S.dx_**2 = }')

        # Use Crank Nicolson to solve
        S.forward_scheme(method="Crank Nicolson", plot=True, animate=False, path="output/task_3/t39_delta_comp_211.pdf")

        # With delta inital compared to 2.11
        S = Schrodinger(Nx=NX, pot_type="barrier", v0=0, Nt=40, T=2e-7)
        S.eigen()
        S.init_cond(name="delta")
        # Discretize t again
        S.discretize_x_t()
        print(f'{S.dt_/S.dx_**2 = }')

        # Use Crank Nicolson to solve
        S.forward_scheme(method="Crank Nicolson", plot=True, animate=False, path="output/task_3/t39_delta_comp_211.pdf")
        
        t2 = time.time(); print(f'Task 3.9 time: {t2 - t1:.4e}')

    
def Task_4():

    if TASK_4_1:
        t1 = time.time(); print("\nTask 4.1")

        v0 = 100
        vr_low = -1e3
        vr_high = 1e3
        # vr_low = -100
        # vr_high = 100
        S = Schrodinger(Nx=NX, pot_type="detuning", v0=v0, Nt=10, vr=0)
        S.eigen()
        S.init_cond(name="eigenfuncs", eigenfunc_idxs=[0])
        S.plot_Psi_0()

        # Energy difference
        print(f'epsilon_0 = {S.eig_vals[1] - S.eig_vals[0]}')

        # Plot vr dependence
        fig = S.detuning_Vr_dependence(vr_low=vr_low, vr_high=vr_high)
        
        t2 = time.time(); print(f'Task 4.1 time: {t2 - t1:.4e}')

    
    if TASK_4_2:
        t1 = time.time(); print("\nTask 4.2")

        # Find tunneling amplitude as function of vr
        S.tunneling_amplitude(vr_low=vr_low, vr_high=vr_high, N=101, fig=fig)
        # tau(vr) = -0.3140942438188899 * vr + -1.0424101057055188

        t2 = time.time(); print(f'Task 4.2 time: {t2 - t1:.4e}')

    
    if TASK_4_4:
        t1 = time.time(); print("\nTask 4.4")

        v0 = 100
        S = Schrodinger(Nx=NX, pot_type="detuning", v0=v0, Nt=1000, vr=0)
        
        # Calculate a reasonable time
        S.eigen()
        epsilon_0 = S.eig_vals[1] - S.eig_vals[0]
        tau = 0.02 * epsilon_0
        # S.T = 12*np.pi*hbar/tau
        S.T = 12*np.pi*hbar/tau
        
        # Discretize again
        S.discretize_x_t()
        S.discretize_pot()

        # Run function
        S.Rabi_oscillations()
        
        t2 = time.time(); print(f'Task 4.4 time: {t2 - t1:.4e}')

    
if __name__ == '__main__':
    # Create dirs
    for i in range(2,5):
        if not os.path.exists(f"output/task_{i}/"):
            os.makedirs(f"output/task_{i}/")

    if TASK_2:
        Task_2()

    if TASK_3:
        Task_3()

    if TASK_4:
        Task_4()



# TODO:
    # Fix same size for Nx in loaded eigvals and Psi_0: is correct for 1000
    # check that disc_x_t comes before
    # Fix potential plot in Forward euler og Crank
    # 3.8-3.9
    # Endre slik at n=0 er laveste state
    # Do error calc with psi and not psi**2
    
    # Figure text equal to Figure caption in report


# Questions:
    # Task 3.3: Initial cond psi n=1,3?
    # Task 3.3: No tunneling
    # Task 3.7: Normalisere etter hver loop?
    # Task 4.4: What is wrong


# Sammenligne res:
    # Task 3.5 - Root values vs eigenvalues
    # 3.8-3.9