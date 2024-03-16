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
    def __init__(self, monomers=10, grid_size=-1,  flexibility=1.0, output_path="output/") -> None:
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
        
        # Grid to store polymer number+1: uint8 = 0-255
        self.monomer_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Ordered list of monomer positions and amino acid number
        self.monomer_pos = np.zeros((monomers, 3), dtype=np.uint16)

        # Initialize polymer
        self.init_polymer()

        # Create monomer-monomer interaction energy matrix
        self.create_interaction_matrix()

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
        print(f'Polymer initialisation used {i}/{tries} tries.')

    def _init_polymer(self):
        """Create initial polymer
        """
        # Clear grid
        self.monomer_grid[:,:] = 0

        # Start the monomer center left (y, x)
        pos = np.ones(2, dtype=np.int16) * ((self.grid_size - self.monomers) // 2)

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
                if self.monomer_grid[pos[0]+direction[0], pos[1]+direction[1]] == 0:
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
            amino_acid_number = np.random.randint(1, 21, dtype=np.uint8)
            self.monomer_grid[pos[0], pos[1]] = i+1 # amino_acid_number
            self.monomer_pos[i,0:2] = pos
            self.monomer_pos[i,2] = amino_acid_number

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

    def create_interaction_matrix(self, plot=False):
        """Create interaction matrix
        """
        # Uniform random matrix from -2kb to -4kb
        self.MM_interaction_energy = np.random.random((20,20))*2 - 4

        # Make it symmetric by copying lower triangle to upper triangle
        self.MM_interaction_energy = np.tril(self.MM_interaction_energy) + np.triu(self.MM_interaction_energy.T, 1)

        # Plot matrix
        if plot:
            fig = plt.figure()
            plt.title("Monomer-Monomer Interaction Energy")
            sn.heatmap(self.MM_interaction_energy, annot=False, vmin=-4, vmax=-2,
                    #    index=r"$Amino acid$", columns=r"$Amino acid$", 
                       cbar_kws={"label": r"Energy $(k_b)$"})
            plt.xlabel("Amino acid")
            plt.ylabel("Amino acid")
            plt.show()
            fig.savefig(self.output_path + "task_1/MM_interaction_energy.pdf")
            plt.close()
        
    def find_nearest_neighbours(self):
        """Generate list of nearesest neighbours for all monomers
        """

        # Pairs of connecting monomers
        # [[monomer5, monomer9],...]
        self.NN = []

        get_surrounding_coords = np.array([[0,1],[1,0],[0,-1],[-1,0]])
        for i in range(self.monomers):
            surrounding_coords = self.monomer_pos[i,:-1] - get_surrounding_coords            

            for coord in surrounding_coords:
                # Skip if previous or next monomer
                # if i < 3:
                #     print(f'{self.monomer_pos[max(0,i-1):min(self.monomers,i+2),:-1] = }')
                #     print(f'{coord = }')
                if (coord == self.monomer_pos[max(0,i-1):min(self.monomers,i+2),:-1]).all(1).any():
                    # if i < 3:
                    #     print("No")
                    continue
                
                # Neighbouring monomer
                if self.monomer_grid[coord[0], coord[1]] != 0:
                    new_connection = [i, self.monomer_grid[coord[0], coord[1]]-1]

                    # Check if opposite is not there already
                    if new_connection[::-1] not in self.NN:
                        self.NN.append(new_connection)
