import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os
from numba import njit

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


class Polymer:
    # Create interaction matrix as static such that all class instances uses the same matrix
    # Uniform random matrix from -2kb to -4kb
    MM_interaction_energy = np.random.random((20,20))*2 - 4
    # Make it symmetric by copying lower triangle to upper triangle
    MM_interaction_energy = np.tril(MM_interaction_energy) + np.triu(MM_interaction_energy.T, 1)

    def __init__(self, monomers=10, grid_size=-1,  flexibility=1.0, output_path="output/", T=10) -> None:
        """Initialize polymer class

        Args:
            monomers (int, optional): number of monomers in chain. Defaults to 10.
            flexibility (float, optional): how straight the monomer is made. 0 becomes straight line. Defaults to 1.0.
            grid_size (int, optional): grid points in each dimension. -1 sets grid_size to monomers*2. Defaults to 20.
        """
        # Set up directories
        self.dir_setup(output_path)

        self.monomers = monomers
        self.flexibility = flexibility
        self.grid_size = grid_size if grid_size > -1 else 2 * monomers
        self.T = T
        self.beta = 1 / T # 1/TkB: kb=1
        
        # Grid to store polymer number+1: uint8 = 0-255
        self.monomer_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8) - 1

        # Ordered list of monomer positions and amino acid number
        self.monomer_pos = np.zeros((monomers, 3), dtype=np.uint16)

        # Initialize polymer
        self.init_polymer()


    def dir_setup(self, output_path):
        self.output_path = output_path 
        if output_path[-1] != "/":
            output_path += "/"

        if not os.path.exists(output_path + "task_1/"):
            os.makedirs(output_path + "task_1/")
        if not os.path.exists(output_path + "task_2/"):
            os.makedirs(output_path + "task_2/")

    def init_polymer(self):
        # Returns false if polymer got stuck
        tries = 10
        for i in range(tries):
            if self._init_polymer(): break

        assert i != tries-1, "Polymer initialisation got stuck"
        # print(f'Polymer initialisation used {i}/{tries} tries.')

    def _init_polymer(self):
        """Create initial polymer
        """
        # Clear grid and pos
        self.monomer_grid[:,:] = -1
        self.monomer_pos[:,:] = 0

        # Start the monomer center (y, x)
        pos = np.ones(2, dtype=np.int16) * (self.grid_size // 2)

        # Rotation matrix for direction change
        rotate = np.array([[0, 1], [-1, 0]], dtype=np.int8)

        # Direction to move y, x
        direction = np.array([0, 0], dtype=np.int8)

        # Make start direction random
        xory = 1*(np.random.random() > .5)
        posorneg = 1 - 2*(np.random.random() > .5)
        direction[xory] = posorneg
        
        # Make polymer
        for i in range(self.monomers):
            # Change direction based on flexibility
            # flexibility = probability of not continuing straight
            if np.random.random() < self.flexibility:
                # Rotate 90 deg clockwise or ccw
                if np.random.random() < .5:
                    direction = rotate @ direction
                else:
                    direction = -rotate @ direction

            # Avoid self
            # Random direction of rotation if colliding
            cw = np.random.random() < .5
            for ii in range(4):
                # Direction works
                if self.monomer_grid[pos[0]+direction[0], pos[1]+direction[1]] == -1:
                    break
                # Go to other possible direction
                else:
                    if cw:
                        direction = rotate @ direction
                    else:
                        direction = -rotate @ direction
            
            # No direction possible
            if ii == 3:
                return False

            # Take a step
            pos += direction

            # Create random amino acid and save
            amino_acid_number = np.random.randint(0, 20, dtype=np.uint8)
            self.monomer_grid[pos[0], pos[1]] = i # Sequential monomer number
            self.monomer_pos[i,:-1] = pos
            self.monomer_pos[i,-1] = amino_acid_number

        # Success
        return True

    def plot_polymer(self):
        fig, ax = plt.subplots()

        # Plot polymer line
        plt.plot(self.monomer_pos[:, 0], self.monomer_pos[:, 1], color="k")
        
        # Plot nearest neighbours
        if self.NN is not None:
            for pair in self.NN:
                plt.plot(
                    [self.monomer_pos[pair[0],0], self.monomer_pos[pair[1],0]],
                    [self.monomer_pos[pair[0],1], self.monomer_pos[pair[1],1]],
                    '--', color="r", linewidth=1)
                
        # Show number and use colormap to show which amino acid
        cm = plt.get_cmap('tab20')
        for i in range(self.monomers):
            plt.plot(self.monomer_pos[i, 0], self.monomer_pos[i, 1], 'o', color=cm(self.monomer_pos[i, -1]-1))
            plt.text(self.monomer_pos[i, 0]-0.25, self.monomer_pos[i, 1]+0.1, str(i))

        # Prettify
        ax.grid(True, linestyle='--')
        ax.set_ylim([self.monomer_pos[:,1].min()-0.5, self.monomer_pos[:,1].max()+0.5])
        ax.set_xlim([self.monomer_pos[:,0].min()-0.5, self.monomer_pos[:,0].max()+0.5])
        yticks = np.arange(self.monomer_pos[:,1].min(), self.monomer_pos[:,1].max()+1)
        xticks = np.arange(self.monomer_pos[:,0].min(), self.monomer_pos[:,0].max()+1)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        # Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2.5)
            ax.spines[axis].set_color('black')
        plt.show()
        plt.close()
        plt.clf()
        plt.close()

    def plot_interaction_matrix(self, save=True):
        """Plot interaction matrix
        """
        # Plot matrix
        fig = plt.figure()
        plt.title("Monomer-Monomer Interaction Energy")
        sn.heatmap(self.MM_interaction_energy, annot=False, vmin=-4, vmax=-2,
                    cbar_kws={"label": r"Energy $(k_b)$"})
        plt.xlabel("Amino acid")
        plt.ylabel("Amino acid")
        plt.show()
        if save:
            fig.savefig(self.output_path + "task_1/MM_interaction_energy.pdf")
        plt.close()
    
    def find_nearest_neighbours(self):
        """Generate list of nearesest neighbours for all monomers
        """
        
        self.NN = self.njit_find_nearest_neighbours(self.monomers, self.monomer_pos, self.monomer_grid)

    @staticmethod
    @njit # Compile with Numba
    def njit_find_nearest_neighbours(monomers, monomer_pos, monomer_grid):
        # Pairs of connecting monomers
        # [[monomer5, monomer9],...] or eqv in 3d 
        dim = len(monomer_grid.shape)
        NN = np.array([[0,0]], dtype=np.uint8)
        new_connection = np.array([[0,0]], dtype=np.uint8)

        get_surrounding_coords = np.array([[0,1],[1,0],[0,-1],[-1,0]])
        for i in range(monomers):
            surrounding_coords = monomer_pos[i,:-1] - get_surrounding_coords            

            for coord in surrounding_coords:
                # Skip if previous or next monomer
                if i > 0:
                    if np.array_equal(coord, monomer_pos[i-1,:-1]):
                        continue
                if i < monomers-1:
                    if np.array_equal(coord, monomer_pos[i+1,:-1]):
                        continue
                
                # Neighbouring monomer
                if monomer_grid[coord[0], coord[1]] != -1:
                    # Set new connection
                    new_connection[0,0] = i
                    new_connection[0,1] = monomer_grid[coord[0], coord[1]]

                    # Check if opposite is not there already
                    if ((new_connection[0,::-1]==NN).sum(axis=1)==dim).sum()==0:
                        NN = np.vstack((NN, new_connection))

        return NN[1:]

    def calculate_energy(self):
        """Calculate the energy of the polymer
        """
        self.E = self.njit_calculate_energy(self.NN, self.MM_interaction_energy, self.monomer_pos)

    @staticmethod
    @njit # Compile with numba
    def njit_calculate_energy(NN, MM_interaction_energy, monomer_pos):
        if len(NN)==0:
            return 0
        
        E = 0
        for pair in NN:
            # Find energy between monomer types
            E += MM_interaction_energy[monomer_pos[pair[0],-1], monomer_pos[pair[1],-1]]
        return E

    def init_multiple(self, N=1000, plot=False, bins=100, save=False):

        """Initiate multiple tertary structures and calculate their energies

        Args:
            N (int, optional): Number of polymers to create. Defaults to 10.
            plot (bool, optional): Plot histogram. Defaults to False.
            bins (int, optional): Histogram bins. Defaults to 10.
            save (bool, optional): Save histogram. Defaults to False.
        """
        E_list = np.zeros(N)

        for i in range(N):
            # Create new initial configuration
            self.init_polymer()

            # Find nearest neighbours
            self.find_nearest_neighbours()

            # Calculate energy
            self.calculate_energy()

            # if plot:
            #     self.plot_polymer()

            E_list[i] = self.E

        if plot:
            fig = plt.figure()
            sn.histplot(E_list, bins=bins, color="k")
            plt.xlabel(r"$Energy (k_b)$")
            plt.title("Energy histogram of tertiary structures")
            plt.show()
            if save:
                fig.savefig(self.output_path + "task_1/t13_energy_hist.pdf")

    def MMC(self):
        """Metropolis Monte Carlo
        """
        pass

    @staticmethod
    @njit
    def njit_find_possible_configurations(monomer_grid):
        """Given a grid of monomers, find legal transitions

        Args:
            monomer_grid (np.ndarray): 2d or 3d grid where monomers are 0-19 and empty space is -1
        """

        pass


