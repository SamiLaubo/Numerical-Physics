import numpy as np
import matplotlib.pyplot as plt


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
    def __init__(self, monomers=10, grid_size=-1,  flexibility=1.0) -> None:
        """Initialize polymer class

        Args:
            monomers (int, optional): number of monomers in chain. Defaults to 10.
            flexibility (float, optional): how straight the monomer is made. 0 becomes straight line. Defaults to 1.0.
            grid_size (int, optional): grid points in each dimension. -1 sets grid_size to monomers*2. Defaults to 20.
        """

        if grid_size > -1:
            self.grid_size = grid_size
        else:
            self.grid_size = 2 * monomers

        self.monomers = monomers
        self.flexibility = flexibility
        
        # int8 = 0-255
        self.monomer_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Ordered list of monomer positions and amino acid number
        self.monomer_pos = np.zeros((monomers, 3), dtype=np.uint16)

        # Initialize polymer
        # Returns false if polymer got stuck
        tries = 10
        for i in range(tries):
            if self.init_polymer(): break

        assert i != tries-1, "Polymer initialisation got stuck"
        print(f'Polymer initialisation used {i}/{tries} tries.')
        

    # Create initial monomer on grid
    def init_polymer(self):
        # Clear grid
        self.monomer_grid[:,:] = 0

        # Start the monomer center left (y, x)
        pos = np.ones(2, dtype=np.int16) * ((self.grid_size - self.monomers) // 2)

        # Rotation matrix for direction change
        rotate = np.array([[0, 1], [-1, 0]], dtype=np.int8)

        # Direction to move y, x
        direction = np.array([0, 0], dtype=np.int8)
        temp_direction = np.array([0, 0], dtype=np.int8)

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
            self.monomer_grid[pos[0], pos[1]] = amino_acid_number
            self.monomer_pos[i,0:2] = pos
            self.monomer_pos[i,2] = amino_acid_number


        # Success
        return True

    def plot_polymer(self):
        fig, ax = plt.subplots()

        # Plot polymer line
        plt.plot(self.monomer_pos[:, 0], self.monomer_pos[:, 1], color="k")
        
        # Show number and use colormap to show which amino acid
        cm = plt.get_cmap('tab20')
        for i in range(self.monomers):
            plt.plot(self.monomer_pos[i, 0], self.monomer_pos[i, 1], 'o', color=cm(self.monomer_pos[i, 2]-1))
            plt.text(self.monomer_pos[i, 0]-0.25, self.monomer_pos[i, 1]+0.1, str(i))

        # Prettyfy
        ax.grid(True, linestyle='--')
        ax.set_yticks(np.arange(int(ax.get_ylim()[0]), np.ceil(ax.get_ylim()[1])))
        ax.set_xticks(np.arange(int(ax.get_xlim()[0]), np.ceil(ax.get_xlim()[1])))
        ax.set_ylim([self.monomer_pos[:,1].min()-0.5, self.monomer_pos[:,1].max()+0.5])
        ax.set_xlim([self.monomer_pos[:,0].min()-0.5, self.monomer_pos[:,0].max()+0.5])
        ax.set_aspect("equal")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2.5)
            ax.spines[axis].set_color('black')
        plt.show()
