import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.sparse.linalg import eigsh

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
def create_V(N, type="random"):
    """Create state vector for system.

    Args:
        N (int): Number of nodes
        type (str): Type of initialization
    """

    # Create random vector and normalize
    if type == "random":
        V = np.random.random(N) / N
    elif type=="Gaussian":
        V = np.exp(-(np.arange(N) - N//2)**2/3)/N

    return V

# 2.2.b
def evolve_VT(V, T, N=1, plot_idx=None, path="", axs=None, titles=None):
    """Evolve network

    Args:
        V (np.ndarray[N]): State vector
        T (np.ndarray[N,N]): Transformation matrix
        N (int, optional): Steps to take
        plot_idx (list): List of step indices to plot. 0 is after first step
        axs (list): plt.Axes to do plots on. If None create new axis
    """

    if plot_idx is not None:
        if axs is None:
            fig, axs = plt.subplots(len(plot_idx)+1, 1, sharex=True, figsize=(5,10))
            axs = axs.ravel()

        axs[0].plot(V, "-o", color="k")
        if titles in not None:
            axs[0].set_title(f"{titles[0]} initial state")
        else:
            axs[0].set_title("Initial state")
        axs[0].grid(False)
        axs[0].set_ylim([0,0.05])
        axs[0].set_ylabel(r"Charge [$Q$]")
        
        axs_idx = 1

    for i in range(N):
        # One step
        V = T@V

        if plot_idx is not None:
            if i in plot_idx:
                # Plot state as line
                axs[axs_idx].plot(V, "-o", color="k")

                axs[axs_idx].set_title(f"Step {i+1}")
                axs[axs_idx].grid(False)
                axs[axs_idx].set_ylim([0,0.05])
                axs[axs_idx].set_ylabel(r"Charge [$Q$]")

                if axs_idx == len(plot_idx):
                    if titles is not None:
                        axs[axs_idx].set_xlabel(f"Node index\n{titles[1]}")
                    else:
                        axs[axs_idx].set_xlabel("Node index")


                axs_idx += 1

    if plot_idx is not None and axs is None:
        plt.tight_layout()
        plt.show()

        if len(path) > 0:
            fig.savefig(path)

    return V


def angle(vec1, vec2):
    """Return angle in radians between two vectors

    Args:
        vec1 (np.ndarray[N]): Vector 1
        vec2 (np.ndarray[N]): Vector 2
    """

    # Unit vectors
    vec1_unit = vec1 / np.linalg.norm(vec1)
    vec2_unit = vec1 / np.linalg.norm(vec1)
    return np.arccos(np.clip(vec1_unit @ vec2_unit, -1.0, 1.0)) # Bc. of numerical inaccuracies in this case

# 2.2.c
def eigvals(T):
    """Calculate eigenvalues and eigenvectors of T with iterative method.

    Args:
        T (np.ndarray[N,N]): Transformation matrix
    """

    # Scipy with Lanczov method
    # Calculates biggest and smallest in magnitude so need to check multiple
    # because of negative eigenvalues
    eig_val_max, eig_vec_max = eigsh(T, which="LM", k=3)
    eig_val_min, eig_vec_min = eigsh(T, which="SM", k=3)

    print("Iterative with Lanczov method:")
    print(f'\tLargest magnitude eigenvalues: {eig_val_max}')
    print(f'\tSmallest magnitude eigenvalues: {eig_val_min}')

    # Numpy version - not using iterative method
    np_eigval, np_eigvec = np.linalg.eigh(T)

    # Find three largest and smallest eigenvalues in magnitude
    np_eigval_argsort = np.argsort(np.abs(np_eigval))

    print("\nWith numpy linalg:")
    print(f'\tLargest magnitude eigenvalue: {np_eigval[np_eigval_argsort[-3:]]}')
    print(f'\tSmallest magnitude eigenvalue: {np_eigval[np_eigval_argsort[:3]]}')
    
    # Dot product of eigenvectors to meassure similarity
    print("\nEigenvectors:")
    print("\tLargest magnitude eigenvalue:")
    print(f'\tAngle between Lanczov and linalg: {angle(eig_vec_max[:, -1], np_eigvec[:, np_eigval_argsort[-1]]):.5f} rad')
    
    print("\n\tSmallest magnitude eigenvalue:")
    print(f'\tAngle between Lanczov and linalg (1): {angle(eig_vec_max[:, -1], np_eigvec[:, np_eigval_argsort[0]]):.5f} rad')
    print(f'\tAngle between Lanczov and linalg (2): {angle(eig_vec_max[:, -2], np_eigvec[:, np_eigval_argsort[1]]):.5f} rad')

    print(f'\n\nAll eigenvalues (linalg): {np_eigval}\n')