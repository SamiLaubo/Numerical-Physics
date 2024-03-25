import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os
from numba import njit
from itertools import product
from tqdm import tqdm

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

    def __init__(self, monomers=10, grid_size=-1,  flexibility=1.0, output_path="output/", T=10, dims=2) -> None:
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
        self.grid_size = grid_size if grid_size > -1 else (5 * monomers) // 2
        self.T = T
        self.beta = 1 / T # 1/TkB: kb=1
        self.dims = dims # 2d or 3d
        
        # Grid to store monomer indexes: int16 max 32767/2 monomers
        self.monomer_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int16) - 1

        # Ordered list of monomer positions
        self.monomer_pos = np.zeros((monomers, self.dims), dtype=np.int16)

        # Ordered list with monomer amino acid number
        self.monomer_AA_number = np.zeros(monomers, dtype=np.uint8)

        # Initialize polymer
        self.init_polymer()

        # Other attribs
        self.NN = None
        self.E = None

        # Direction to surrounding coordinates in 2d and 3d with "corners"
        self.surrounding_coords = np.asarray(list(product([-1,0,1], repeat=self.dims)), dtype=np.int8)
        # Remove origin (0,0(,0))
        self.surrounding_coords = self.surrounding_coords[~((self.surrounding_coords**2).sum(axis=1)==0),:]


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
        self.monomer_grid[:] = -1
        self.monomer_pos[:] = 0
        self.monomer_AA_number[:] = 0

        # Start the monomer center (y, x, [z])
        pos = np.ones(self.dims, dtype=np.int16) * (self.grid_size // 2)

        # Rotation matrix for direction change
        if self.dims == 2:
            rotate = np.array([[0, 1], [-1, 0]], dtype=np.int8)
        else:
            rotate = np.array([[0, 1], [-1, 0]], dtype=np.int8)
            print(f'Dimension 3d is not implemented in _init_polymer')

        # Direction to move y, x [,z]
        direction = np.zeros(self.dims, dtype=np.int8)

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
            # amino_acid_number = np.random.randint(0, 20, dtype=np.uint8)
            self.monomer_grid[pos[0], pos[1]] = i # Sequential monomer number
            self.monomer_pos[i] = pos
            # self.monomer_pos[i,-1] = amino_acid_number
            self.monomer_AA_number[i] = np.random.randint(0, 20, dtype=np.uint8)

        # Success
        return True
    
    def remember_initial(self):
        self.init_monomer_grid = self.monomer_grid.copy()
        self.init_monomer_pos = self.monomer_pos.copy()

    def reset_to_initial(self):
        self.monomer_grid = self.init_monomer_grid.copy()
        self.monomer_pos = self.init_monomer_pos.copy()

        self.find_nearest_neighbours()
        self.calculate_energy()


    def plot_polymer(self, MC_step=-1, ax=None):
        show = False
        if ax is None:
            show = True
            fig, ax = plt.subplots()

        if MC_step > -1:
            ax.set_title(f"MC Step {MC_step}")

        # Plot polymer line
        ax.plot(self.monomer_pos[:, 0], self.monomer_pos[:, 1], color="k")
        
        # Plot nearest neighbours
        if self.NN is not None:
            for pair in self.NN:
                try:
                    ax.plot(
                        [self.monomer_pos[pair[0],0], self.monomer_pos[pair[1],0]],
                        [self.monomer_pos[pair[0],1], self.monomer_pos[pair[1],1]],
                        '--', color="r", linewidth=1)
                except:
                    print(f"Failed at pair {pair}")
                
        # Show number and use colormap to show which amino acid
        cm = plt.get_cmap('tab20')
        for i in range(self.monomers):
            ax.plot(self.monomer_pos[i, 0], self.monomer_pos[i, 1], 'o', color=cm(self.monomer_AA_number[i]))
            ax.text(self.monomer_pos[i, 0]-0.25, self.monomer_pos[i, 1]-0.25, str(i))

        # Prettify
        ax.grid(True, linestyle='--')
        ax.set_ylim([self.monomer_pos[:,1].max()+0.5, self.monomer_pos[:,1].min()-0.5])
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
        
        if show:
            plt.show()
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
            surrounding_coords = monomer_pos[i] - get_surrounding_coords            

            for coord in surrounding_coords:
                # if i > 0:
                #     if np.array_equal(coord, monomer_pos[i-1]):
                #         continue
                # if i < monomers-1:
                #     if np.array_equal(coord, monomer_pos[i+1]):
                #         continue
                # Skip if previous or next monomer

                # Only if not empty and not next or previous in covalent bonds
                if monomer_grid[coord[0], coord[1]] not in [-1, i-1, i+1]:
                
                # Neighbouring monomer
                # if monomer_grid[coord[0], coord[1]] != -1:
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
        self.E = self.njit_calculate_energy(self.NN, self.MM_interaction_energy, self.monomer_AA_number)

    @staticmethod
    @njit # Compile with numba
    def njit_calculate_energy(NN, MM_interaction_energy, monomer_AA_number):
        if len(NN)==0:
            return 0
        
        E = 0
        for pair in NN:
            # Find energy between monomer types
            E += MM_interaction_energy[monomer_AA_number[pair[0]], monomer_AA_number[pair[1]]]
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


    def MMC(self, MC_steps=1, use_threshold=True, threshold=1e-2,  N_thr=3, N_avg=100):
        if self.NN is None:
            self.find_nearest_neighbours()
        if self.E is None:
            self.calculate_energy()

        self.MMC_observables = self.njit_MMC(
                                    self.monomers, self.E, self.T, MC_steps,
                                    self.surrounding_coords, self.MM_interaction_energy,
                                    self.monomer_grid, self.monomer_pos, self.monomer_AA_number, self.NN,
                                    self.njit_find_possible_transitions,
                                    self.njit_apply_transition,
                                    self.njit_find_nearest_neighbours,
                                    self.njit_calculate_energy,
                                    use_threshold, threshold, N_thr, N_avg)
        
        # Update enrgy. Rest is updated by reference in function
        self.E = self.MMC_observables.get("E")[-1]

    @staticmethod
    @njit()
    def njit_MMC(
        monomers, E, T, MC_steps,
        surrounding_coords, MM_interaction_energy,
        monomer_grid, monomer_pos, monomer_AA_number, NN,
        njit_find_possible_transitions,
        njit_apply_transition,
        njit_find_nearest_neighbours,
        njit_calculate_energy,
        use_threshold=True, threshold=1e-2,  N_thr=3, N_avg=100):
        """Metropolis Monte Carlo
        """

        MC_steps = int(MC_steps)

        MMC_observables = {
            "E": np.zeros(MC_steps),
            "e2e": np.zeros(MC_steps),
            "RoG": np.zeros(MC_steps)
        }
        
        # For all steps/sweeps
        running_mean = np.zeros(MC_steps//N_avg)
        # while i < MC_steps and abs(slope) < threshold: 
        # while i < MC_steps and 
        for i in range(MC_steps):
            # Do N (#monomer) draws/trials
            for _ in range(monomers):
                # Draw random monomer
                monomer_idx = np.random.randint(0, monomers)

                # Find possible transitions
                possible_transitions = njit_find_possible_transitions(monomer_grid, monomer_pos, monomer_idx, surrounding_coords)

                # No possible transitions -> continue
                if len(possible_transitions) == 0: continue
                
                # Pick random transition and apply
                transition_idx = np.random.randint(0, possible_transitions.shape[0])
                njit_apply_transition(monomer_grid, monomer_pos, monomer_idx, possible_transitions[transition_idx])

                # Find nearest neighbours
                old_NN = NN.copy()
                NN = njit_find_nearest_neighbours(monomers, monomer_pos, monomer_grid)
                
                # Calculate new energy
                old_E = E
                E = njit_calculate_energy(NN, MM_interaction_energy, monomer_AA_number)

                # Accept based on Metropolis MC rule (or change back)
                acceptance_prob = np.exp(-(E-old_E)/T) if E-old_E > 0 else 1.0
                if acceptance_prob < np.random.random(): # Change back to original (else new state is kept)
                    E = old_E
                    NN = old_NN
                    njit_apply_transition(monomer_grid, monomer_pos, monomer_idx, -possible_transitions[transition_idx])

            # Save observables
            # Energy
            MMC_observables["E"][i] = E
            # End to end euclidean distance
            MMC_observables["e2e"][i] = np.sqrt(((monomer_pos[0] - monomer_pos[-1])**2).sum())
            # Radius of gyration
            center_of_mass = monomer_pos.sum(axis=0)/monomers
            MMC_observables["RoG"][i] = np.sqrt(((monomer_pos-center_of_mass)**2).sum(axis=1).sum()/monomers)

            if use_threshold:
                if i >= N_avg-1 and (i+1) % N_avg == 0:
                    # np.convolve(MMC_observables["E"], np.ones(N_avg)/N_avg, mode="valid")
                    # slope = np.polyfit(np.arange(len(i+1)), MMC_observables.get("E"), 1)[0]
                    # print(slope)
                    running_mean[i//N_avg] = np.mean(MMC_observables.get("E")[i+1-N_avg:i+1])
                    
                    # print(running_mean)

                    # Break if under threshold
                    if i//N_avg >= N_thr:
                        if np.sum((running_mean[i//N_avg-N_thr:i//N_avg+1] - np.mean(running_mean[i//N_avg-N_thr:i//N_avg+1]))**2) < threshold:
                            break

        # Remove unused values
        if use_threshold and i < MC_steps:
            for key, val in MMC_observables.items():
                MMC_observables[key] = val[:i+1]

        return MMC_observables

    @staticmethod
    @njit
    def njit_find_possible_transitions(monomer_grid, monomer_pos, monomer_idx, surrounding_coords):
        """Given a grid of monomers, find legal transitions

        Args:
            monomer_grid (np.ndarray): 2d or 3d grid where monomers are 0-19 and empty space is -1
        """
        possible_transitions = np.zeros((1, monomer_pos.shape[1]), dtype=np.int8)

        if len(monomer_grid.shape) == 2:
            # Check surrounding positions for free space
            for direction in surrounding_coords:
                new_pos = monomer_pos[monomer_idx] + direction
                if monomer_grid[new_pos[0], new_pos[1]] != -1:
                    continue

                # Check if covalent bonds lengths still are 1
                if monomer_idx < monomer_pos.shape[0]-1:
                    # Manhattan distance needs to be 1 to next monomer
                    if np.abs(new_pos - monomer_pos[monomer_idx+1]).sum() != 1:
                        continue
                if monomer_idx > 0:
                    # Manhattan distance needs to be 1 to previous monomer
                    if np.abs(new_pos - monomer_pos[monomer_idx-1]).sum() != 1:
                        continue

                possible_transitions = np.vstack((possible_transitions, direction[None,:]))

            return possible_transitions[1:]    


    @staticmethod
    @njit
    def njit_apply_transition(monomer_grid, monomer_pos, monomer_idx, direction):
        """Apply a transition

        Args:
            monomer_grid (_type_): _description_
            monomer_pos (_type_): _description_
            monomer_idx (_type_): _description_
            direction (_type_): _description_
        """


        # Delete old position
        if len(monomer_pos.shape) == 2:
            monomer_grid[monomer_pos[monomer_idx,0], monomer_pos[monomer_idx,1]] = -1
            
        # Change monomer_pos
        monomer_pos[monomer_idx] += direction

        # Add new position
        if len(monomer_pos.shape) == 2:
            monomer_grid[monomer_pos[monomer_idx,0], monomer_pos[monomer_idx,1]] = monomer_idx

    def plot_MMC(self, running_mean_N=3):
        """Plot observables from MMC
        """

        # Energy
        fig = plt.figure()
        # fig, axs = plt.subplots(3, 2, sharex=True)
        # axs = axs.ravel()
        fig.suptitle(f"Metropolis Monte Carlo\nN = {self.monomers} - T = {self.T:.2f}")

        ax = plt.subplot(321)
        ax.plot(
            np.convolve(self.MMC_observables.get("E"), np.ones(running_mean_N)/running_mean_N, mode="valid"),
            color="k")
        ax.set_title("Energy")
        # ax.set_xlabel("MC Step")
        ax.set_ylabel(r"Energy $(k_b)$")

        # e2e
        ax = plt.subplot(323, sharex=ax)
        ax.plot(
            np.convolve(self.MMC_observables.get("e2e"), np.ones(running_mean_N)/running_mean_N, mode="valid"),
            color="k")
        ax.set_title("End-to-end euclidean distance")
        # ax.set_xlabel("MC Step")
        ax.set_ylabel(r"Distance")
        
        # RoG
        ax = plt.subplot(325, sharex=ax)
        ax.plot(
            np.convolve(self.MMC_observables.get("RoG"), np.ones(running_mean_N)/running_mean_N, mode="valid"),
            color="k")
        ax.set_title("Radius of Gyration")
        ax.set_xlabel("MC Step")
        ax.set_ylabel(r"RoG")

        ax = plt.subplot(122)
        self.plot_polymer(ax=ax, MC_step=len(self.MMC_observables.get("RoG")))
        
        plt.tight_layout()
        plt.show()

    
    def MMC_time_to_equilibrium(self, T_low, T_high, N, max_MC_steps=1e5, threshold=1e-1, N_thr=5, N_avg=100):
        self.remember_initial()

        steps_needed = []
        temps = np.linspace(T_low, T_high, N)

        for T in tqdm(temps):
            # Set back to initial state
            self.reset_to_initial()
            self.T = T

            # Do Monte Carlo
            self.MMC(MC_steps=max_MC_steps, use_threshold=True, threshold=threshold, N_thr=N_thr, N_avg=N_avg)

            # Save amount of steps
            steps_needed.append(len(self.MMC_observables.get("E")))

            self.plot_MMC(running_mean_N=10)

        plt.figure()
        plt.plot(temps, steps_needed)
        plt.title("MC Steps before equilibration")
        plt.xlabel(r"Temperature (k_b)")
        plt.ylabel("Steps")
        plt.show()
            