import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
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
        if dims == 2:
            self.monomer_grid = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.int16) - 1
        else:
            self.monomer_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.int16) - 1

        # Ordered list of monomer positions
        self.monomer_pos = np.zeros((monomers, 3), dtype=np.int16)

        # Ordered list with monomer amino acid number
        self.monomer_AA_number = np.zeros(monomers, dtype=np.uint8)

        # Initialize polymer
        self.init_polymer()

        # Other attribs
        self.NN = None
        self.E = None

        # Direction to surrounding coordinates in 2d and 3d with "corners"
        self.surrounding_coords = np.asarray(list(product([-1,0,1], repeat=self.dims)), dtype=np.int16)
        # Remove origin (0,0(,0))
        self.surrounding_coords = self.surrounding_coords[~((self.surrounding_coords**2).sum(axis=1)==0),:]
        # Add empty axis if 2d
        if self.dims == 2:
        #     self.surrounding_coords = self.surrounding_coords[...,None]
            self.surrounding_coords = np.hstack((self.surrounding_coords, np.zeros((self.surrounding_coords.shape[0],1), dtype=np.int16)))

        # Without corners
        self.surrounding_coords_cross = self.surrounding_coords[((self.surrounding_coords**2).sum(axis=1)==1),:]


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

    def _init_polymer(self):
        """Create initial polymer
        """
        # Clear grid and pos
        self.monomer_grid[:] = -1
        self.monomer_pos[:] = 0
        self.monomer_AA_number[:] = 0

        # Start the monomer center (x, y [,z])
        pos = np.ones(3, dtype=np.int16) * (self.grid_size // 2)

        if self.dims == 2:
            pos[-1] = 0

        # Rotation matrix for direction change
        if self.dims == 2:
            rotate = np.array([[0,1,0],[-1,0,0],[0,0,0]], dtype=np.int8)
        else:
            Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])
            Ry = np.array([[0,0,1],[0,1,0],[-1,0,0]])
            Rz = np.array([[0,-1,0],[1,0,0],[0,0,1]])

            rotate = [Rx, Ry, Rz]

        # Direction to move x, y [,z]
        direction = np.zeros(3, dtype=np.int8)

        # Make start direction random
        direction[np.random.randint(0, self.dims)] = 1 if np.random.random() > .5 else -1

        # Make polymer
        for i in range(self.monomers):
            # Store original direction for 3d rotations
            org_direction = np.where(direction**2==1)[0][0]

            # Change direction based on flexibility
            # flexibility = probability of not continuing straight
            if np.random.random() < self.flexibility:
                if self.dims == 2:
                    # Rotate 90 deg clockwise or ccw
                    if np.random.random() < .5:
                        direction = rotate @ direction
                    else:
                        direction = -rotate @ direction
                else: # 3d
                    # Choose random axis to flip around
                    rand_axis = np.random.random(3)

                    # Want new direction orthogonal to current
                    rand_axis[0] = 0

                    # Axis to flip around
                    flip_axis = np.argmax(rand_axis)

                    # Random sign
                    if np.random.random() < .5:
                        direction = rotate[flip_axis] @ direction
                    else:
                        direction = -rotate[flip_axis] @ direction

            # Avoid self
            # Random direction of rotation if colliding
            cw = np.random.random() < .5
            for ii in range(4):
                # Direction works
                if self.monomer_grid[tuple(pos+direction)] == -1:
                    break

                # Go to other possible direction
                else:
                    if cw:
                        if self.dims == 2:
                            direction = rotate @ direction
                        else:
                            direction = rotate[org_direction] @ direction # Rotate around original direction axis
                    else:
                        if self.dims == 2:
                            direction = -rotate @ direction
                        else:
                            direction = -rotate[org_direction] @ direction
            
            # No direction possible
            if ii == 3:
                return False

            # Take a step
            pos += direction

            # Create random amino acid and save
            self.monomer_grid[pos[0], pos[1], pos[2]] = i # Sequential monomer number
            self.monomer_pos[i] = pos
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
            if self.dims == 2:
                fig, ax = plt.subplots()
            else:
                fig = plt.figure()
                ax = plt.axes(projection="3d")

        if MC_step > -1:
            ax.set_title(f"MC Step {MC_step}")

        # Plot polymer line
        if self.dims == 2:
            ax.plot(self.monomer_pos[:, 0], self.monomer_pos[:, 1], color="k")
        else:
            ax.plot3D(self.monomer_pos[:, 0], self.monomer_pos[:, 1], self.monomer_pos[:, 2], color="k")

        
        # Plot nearest neighbours
        if self.NN is not None:
            for pair in self.NN:
                if self.dims == 2:
                    try:
                        ax.plot(
                            [self.monomer_pos[pair[0],0], self.monomer_pos[pair[1],0]],
                            [self.monomer_pos[pair[0],1], self.monomer_pos[pair[1],1]],
                            '--', color="r", linewidth=1)
                    except:
                        print(f"Failed at pair {pair}")
                else:
                    try:
                        ax.plot3D(
                            [self.monomer_pos[pair[0],0], self.monomer_pos[pair[1],0]],
                            [self.monomer_pos[pair[0],1], self.monomer_pos[pair[1],1]],
                            [self.monomer_pos[pair[0],2], self.monomer_pos[pair[1],2]],
                            '--', color="r", linewidth=1)
                    except:
                        print(f"Failed at pair {pair}")
                
        # Show number and use colormap to show which amino acid
        cm = plt.get_cmap('tab20')
        for i in range(self.monomers):
            if self.dims == 2:
                ax.plot(self.monomer_pos[i, 0], self.monomer_pos[i, 1], 'o', color=cm(self.monomer_AA_number[i]))
                ax.text(self.monomer_pos[i, 0]-0.25, self.monomer_pos[i, 1]-0.25, str(i))
            else:
                ax.plot(self.monomer_pos[i, 0], self.monomer_pos[i, 1], self.monomer_pos[i, 2], 'o', color=cm(self.monomer_AA_number[i]))

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

        if self.dims == 3:
            ax.set_zlim([self.monomer_pos[:,2].max()+0.5, self.monomer_pos[:,2].min()-0.5])
            zticks = np.arange(self.monomer_pos[:,2].min(), self.monomer_pos[:,2].max()+1)
            ax.set_zticks(zticks)
            ax.set_zticklabels([])

        ax.set_aspect("equal")
        # Border
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2.5)
            ax.spines[axis].set_color('black')
        
        if show:
            plt.show()
            plt.close()
        

    def plot_interaction_matrix(self, save=True, path="task_1/MM_interaction_energy.pdf"):
        """Plot interaction matrix
        """
        # Plot matrix
        fig = plt.figure()
        plt.title("Monomer-Monomer Interaction Energy")
        sn.heatmap(self.MM_interaction_energy, annot=False, vmin=np.floor(self.MM_interaction_energy.min()), vmax=np.ceil(self.MM_interaction_energy.max()),
                    cbar_kws={"label": r"Energy $(k_b)$"})
        plt.xlabel("Amino acid")
        plt.ylabel("Amino acid")
        plt.show()
        if save:
            fig.savefig(self.output_path + path)
        plt.close()
    
    def find_nearest_neighbours(self):
        """Generate list of nearesest neighbours for all monomers
        """
        
        self.NN = self.njit_find_nearest_neighbours(self.monomers, self.monomer_pos, self.monomer_grid, self.surrounding_coords_cross)

    @staticmethod
    @njit # Compile with Numba
    def njit_find_nearest_neighbours(monomers, monomer_pos, monomer_grid, surrounding_coords):
        # Pairs of connecting monomers
        # [[monomer5, monomer9],...]
        NN = np.array([[0,0]], dtype=np.uint8)
        new_connection = np.array([[0,0]], dtype=np.uint8)

        for i in range(monomers):
            cur_surrounding_coords = monomer_pos[i] - surrounding_coords      

            for coord in cur_surrounding_coords:
                # Only if not empty and not next or previous in covalent bonds
                if monomer_grid[coord[0], coord[1], coord[2]] not in [-1, i-1, i+1]:
                
                    # Set new connection
                    new_connection[0,0] = i
                    new_connection[0,1] = monomer_grid[coord[0], coord[1], coord[2]]

                    # Check if opposite is not there already
                    if ((new_connection[0,::-1]==NN).sum(axis=1)==2).sum()==0:
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


    def MMC(self, MC_steps=1, use_threshold=True, threshold=1e-2,  N_thr=3, N_avg=100, SA=False):
        if self.NN is None:
            self.find_nearest_neighbours()
        if self.E is None:
            self.calculate_energy()

        self.MMC_observables, self.NN = self.njit_MMC(
                                            self.monomers, self.E, self.T, MC_steps,
                                            self.surrounding_coords, self.surrounding_coords_cross, self.MM_interaction_energy,
                                            self.monomer_grid, self.monomer_pos, self.monomer_AA_number, self.NN,
                                            self.njit_find_possible_transitions,
                                            self.njit_apply_transition,
                                            self.njit_find_nearest_neighbours,
                                            self.njit_calculate_energy,
                                            use_threshold, threshold, N_thr, N_avg,
                                            SA=SA)
        
        # Update enrgy. Rest is updated by reference in function
        self.E = self.MMC_observables.get("E")[-1]

    @staticmethod
    @njit()
    def njit_MMC(
        monomers, E, T, MC_steps,
        surrounding_coords, surrounding_coords_cross, MM_interaction_energy,
        monomer_grid, monomer_pos, monomer_AA_number, NN,
        njit_find_possible_transitions,
        njit_apply_transition,
        njit_find_nearest_neighbours,
        njit_calculate_energy,
        use_threshold, threshold, N_thr, N_avg,
        SA):
        """Metropolis Monte Carlo
        """

        MC_steps = int(MC_steps)

        MMC_observables = {
            "E": np.zeros(MC_steps),
            "e2e": np.zeros(MC_steps),
            "RoG": np.zeros(MC_steps)
            # "NN": np.array([[0,0]], dtype=np.uint8)
        }

        # For all steps/sweeps
        running_mean = np.zeros(MC_steps//N_avg)
        for i in range(MC_steps):
            # Simulated annealing temperature
            if SA:
                T = 3 - 2 * i / MC_steps # T = 3 to T = 1

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
                NN = njit_find_nearest_neighbours(monomers, monomer_pos, monomer_grid, surrounding_coords_cross)
                
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
                    running_mean[i//N_avg] = np.mean(MMC_observables.get("E")[i+1-N_avg:i+1])
                    
                    # Break if under threshold
                    if i//N_avg >= N_thr:
                        if np.sum((running_mean[i//N_avg-N_thr:i//N_avg+1] - np.mean(running_mean[i//N_avg-N_thr:i//N_avg+1]))**2) < threshold:
                            break

        # Remove unused values
        if use_threshold and i < MC_steps:
            for key, val in MMC_observables.items():
                MMC_observables[key] = val[:i+1]

        # Return NN as it is not changed by reference
        # MMC_observables["NN"] = NN

        return MMC_observables, NN

    @staticmethod
    @njit
    def njit_find_possible_transitions(monomer_grid, monomer_pos, monomer_idx, surrounding_coords):
        """Given a grid of monomers, find legal transitions

        Args:
            monomer_grid (np.ndarray): 3d grid where empty space is -1.
        """
        possible_transitions = np.zeros((1, 3), dtype=np.int16)

        # Check surrounding positions for free space
        for direction in surrounding_coords:
            new_pos = monomer_pos[monomer_idx] + direction

            # If empty
            if monomer_grid[new_pos[0], new_pos[1], new_pos[2]] != -1:
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
        monomer_grid[monomer_pos[monomer_idx,0], monomer_pos[monomer_idx,1], monomer_pos[monomer_idx,2]] = -1
            
        # Change monomer_pos
        monomer_pos[monomer_idx] += direction

        # Add new position
        monomer_grid[monomer_pos[monomer_idx,0], monomer_pos[monomer_idx,1], monomer_pos[monomer_idx,2]] = monomer_idx

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

        if self.dims == 2:
            ax = plt.subplot(122)
        else:
            ax = plt.subplot(122, projection="3d")
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
        plt.xlabel(r"Temperature $(k_b)$")
        plt.ylabel("Steps")
        plt.show()
            