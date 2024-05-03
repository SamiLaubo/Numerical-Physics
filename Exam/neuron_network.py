import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import scipy
import time

# 2.2.a
def create_T(N_nodes, N_neighbours, plot=False, path=""):
    """Create transformation matrix for N_nodes nodes with N_neighbours closest neighbours in each direction.
        Periodic boundary conditions.


    Args:
        N_nodes (int): Total number of nodes
        N_neighbours (int): Number of closest neighbours for each node en each direction.
    """
    # Create T
    T = np.zeros((N_nodes,N_nodes))

    # Fill off-diagonals and "corners"
    for i in range(1, N_neighbours+1):
        # Off-diagonals
        np.fill_diagonal(T[i:], 1)
        np.fill_diagonal(T[:, i:], 1)

        # Corners
        np.fill_diagonal(T[-i:], 1)
        np.fill_diagonal(T[:, -i:], 1)

    # Normalize
    T /= (2*N_neighbours)

    # Plot matrix
    if plot:
        fig = plt.figure()
        sn.heatmap(T, annot=False)
        plt.xlabel("Node index")
        plt.ylabel("Node index")
        plt.show()
        if len(path) > 0:
            fig.savefig(path)
        plt.close()

    return T

# 2.2.b
def create_V(N, type="random", normalize=True):
    """Create state vector for system.

    Args:
        N (int): Number of nodes
        type (str): Type of initialization
    """

    # Create random vector or gaussian vector
    if type == "random":
        V = np.random.random(N)
    elif type == "Gaussian":
        V = np.exp(-(np.arange(N) - N//2)**2/3)
    elif type == "inv_step":
        V = 1 / (np.arange(N) + 1)

    # Normalize
    if normalize:
        V /= np.sum(V)
    return V

# 2.2.b
def evolve_VT(V, T, N=1, forward=True, method="np.linalg.solve", plot_idx=None, path="", axs=None, titles=None, use_lim=True, figsize=(5,10)):
    """Evolve network

    Args:
        V (np.ndarray[N]): State vector
        T (np.ndarray[N,N]): Transformation matrix
        N (int): Steps to take
        method (str): Method to solve linear system
        plot_idx (list): List of step indices to plot. 0 is after first step
        axs (list): plt.Axes to do plots on. If None create new axis
    """

    solver_time = []
    if method == "scipy.sparse.linalg.spsolve":
        T = scipy.sparse.csc_matrix(T)

    if plot_idx is not None:
        show_plot = False
        if axs is None:
            show_plot = True
            fig, axs = plt.subplots(len(plot_idx)+1, 1, sharex=True, figsize=figsize)
            axs = axs.ravel()

        axs[0].plot(V, "-o", color="k")
        if titles is not None:
            axs[0].set_title(f"{titles[0]} initial state")
        elif forward == False:
            axs[0].set_title(f"Initial state - {method}")
        else:
            axs[0].set_title("Initial state")
        axs[0].grid(False)
        if use_lim:
            axs[0].set_ylim([-0.04,0.4])
        axs[0].set_ylabel(r"Charge [C]")
        
        axs_idx = 1

    for i in range(N):
        # One step
        if forward:
            V = T@V
        else: # Backward
            if method == "np.linalg.solve":
                t1 = time.perf_counter_ns()
                V = np.linalg.solve(T, V)
                t2 = time.perf_counter_ns()
            elif method == "scipy.linalg.solve":
                t1 = time.perf_counter_ns()
                V = scipy.linalg.solve(T, V, assume_a="sym")
                t2 = time.perf_counter_ns()
            elif method == "scipy.sparse.linalg.spsolve":
                t1 = time.perf_counter_ns()
                V = scipy.sparse.linalg.spsolve(T, V)
                t2 = time.perf_counter_ns()
            solver_time.append(t2-t1)

        if plot_idx is not None:
            if i in plot_idx:
                # Plot state as line
                axs[axs_idx].plot(V, "-o", color="k")

                if forward:
                    axs[axs_idx].set_title(f"t = {i+1}dt")
                else:
                    axs[axs_idx].set_title(f"t = -{i+1}dt")

                axs[axs_idx].grid(False)
                if use_lim:
                    axs[axs_idx].set_ylim([-0.04,0.4])
                axs[axs_idx].set_ylabel(r"Charge [C]")

                if axs_idx == len(plot_idx):
                    if titles is not None:
                        axs[axs_idx].set_xlabel(f"Node index\n{titles[1]}")
                    else:
                        axs[axs_idx].set_xlabel("Node index")

                axs_idx += 1

    if plot_idx is not None:
        if show_plot:
            plt.tight_layout()
            plt.show()

        if len(path) > 0:
            fig.savefig(path)

    if forward == False:
        print(f'Time for solver {method} per run ({N=}): {np.mean(solver_time)*1e-6:.5f} ms +- {np.std(solver_time)*1e-6:.5f} ms')

    return V


def angle(vec1, vec2):
    """Return angle in radians between two vectors

    Args:
        vec1 (np.ndarray[N]): Vector 1
        vec2 (np.ndarray[N]): Vector 2
    """

    # Unit vectors
    vec1_unit = vec1 / np.linalg.norm(vec1)
    vec2_unit = vec2 / np.linalg.norm(vec2)
    return np.arccos(np.clip(vec1_unit @ vec2_unit, -1.0, 1.0)) # Bc. of numerical inaccuracies in this case

# 2.2.c
def eigvals(T, verbal=True):
    """Calculate eigenvalues and eigenvectors of T with iterative method.

    Args:
        T (np.ndarray[N,N]): Transformation matrix
    """

    # Scipy with Lanczov method
    # Calculates biggest and smallest in magnitude so need to check multiple
    # because of negative eigenvalues
    t1 = time.perf_counter_ns()
    eig_val_max, eig_vec_max = scipy.sparse.linalg.eigsh(T, which="LM", k=3)
    eig_val_min, eig_vec_min = scipy.sparse.linalg.eigsh(T, which="SM", k=3)
    t2 = time.perf_counter_ns()
    print(f'Lanczov time: {(t2 - t1)*1e-6} ms')

    # Numpy version - not using iterative method
    t3 = time.perf_counter_ns()
    np_eigval, np_eigvec = np.linalg.eigh(T)
    t4 = time.perf_counter_ns()
    print(f'Numpy linalg time: {(t4 - t3)*1e-6} ms')

    # Find three largest and smallest eigenvalues in magnitude
    np_eigval_argsort = np.argsort(np.abs(np_eigval))

    if verbal:
        print("Iterative with Lanczov method:")
        print(f'\tLargest magnitude eigenvalues: {eig_val_max}')
        print(f'\tSmallest magnitude eigenvalues: {eig_val_min}')

        print("\nWith numpy linalg:")
        print(f'\tLargest magnitude eigenvalue: {np_eigval[np_eigval_argsort[-3:]]}')
        print(f'\tSmallest magnitude eigenvalue: {np_eigval[np_eigval_argsort[:3]]}')
        
        # Dot product of eigenvectors to meassure similarity
        print("\nEigenvectors:")
        print("\tLargest magnitude eigenvalue:")
        print(f'\tAngle between Lanczov and linalg: {angle(eig_vec_max[:, -1], np_eigvec[:, np_eigval_argsort[-1]]):.5f} rad')
        print(f'\tAngle between Lanczov and linalg: {angle(eig_vec_max[:, -2], np_eigvec[:, np_eigval_argsort[-1]]):.5f} rad')
        
        print("\n\tSmallest magnitude eigenvalue:")
        print(f'\tAngle between Lanczov and linalg (1): {angle(eig_vec_min[:, -1], np_eigvec[:, np_eigval_argsort[0]]):.5f} rad')
        print(f'\tAngle between Lanczov and linalg (2): {angle(eig_vec_min[:, -2], np_eigvec[:, np_eigval_argsort[1]]):.5f} rad')

        print(f'\n\nAll eigenvalues (linalg): {np_eigval}\n')

    return np_eigval, np_eigvec

# 2.2.c
def join_networks(n1, n2, normalize=True, plot=False, path=""):
    """Joins two network matrices

    Args:
        n1 (np.ndarray[N1,N1]): Unnormalized matrix 1
        n2 (np.ndarray[N2,N2]): Unnormalized matrix 2
    """

    N1 = n1.shape[0]
    N2 = n2.shape[0]
    N = N1 + N2
    T = np.zeros((N,N))

    T[:N1,:N1] = n1
    T[N1:,N1:] = n2

    if normalize:
        T /= np.sum(n1[0,:].sum())

    # Plot matrix
    if plot:
        fig = plt.figure()
        sn.heatmap(T, annot=False)
        plt.xlabel("Node index")
        plt.ylabel("Node index")
        plt.show()
        if len(path) > 0:
            fig.savefig(path)
        plt.close()

    return T