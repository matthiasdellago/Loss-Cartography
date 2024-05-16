#cartographer.py
"""
Cartographer class
Takes   - a model
        - a dataset
        - a loss function

Returns a map of the loss landscape in the parameter space around the model's current parameters (location).
Since the parameter space is so high dimensional, the cartographer resticts itself to a few loss profiles measured in a few directions.
The directions are randomly chosen unless otherwise specified.

The resolution of the profiles is logarithmic in the distance from the current location.
This allows the cartographer to perform a multi-scale analysis of the loss landscape.
In addition to the profiles, the cartographer also performs a roughness analysis of the loss landscape.
For each point on the profile, it measures the roughness at that scale by way of the coastline paradox.
TODO: Distinguish this method from FFT and the Hessian. Explain downsides of each with the counter example.
"""

import torch
from torch import nn
from torch import jit
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.loss import _Loss
from torch.nn import ModuleDict
from loss_locus import LossLocus # vector operations on nn.Module
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from warnings import warn
import plotly.graph_objects as go

class Cartographer:
    """
    A class that analyzes the loss landscape around a machine learning model.

    The Cartographer class takes a model, a dataset, and a loss function, and
    performs a multi-scale analysis of the loss landscape around the model's
    current parameters. It generates a set of random, normalized directions in
    the parameter space and measures the loss profiles along those directions.

    The analysis is performed ona a logarithmic scale, allowing the Cartographer
    to capture both coarse and fine-grained details of the loss landscape.
    This is used to compute the roughness of the loss landscape at different scales.

    It's purpose is to:
        - Provide a visual representation of the loss landscape around a model.
        - Identify distinguished scales in the loss landscape, to aid in learning rate selection.
        - Measure the changes in the scale-dependent roughness, during training.
        - Identify anisotropic roughness in parameter space. Different layers may differ in roughness.
        - Identify inhomogeneities in the loss landscape. Tained models may differ in roughness, from random points.

    Attributes:
        center (LossLocus): The machine learning model to be analyzed.
        dataset (Dataset): The dataset used to compute the loss. For deterministic loss we iterate over the entire dataset. Make sure to use a small dataset.
        loss_function (_Loss): The loss function used to evaluate the model.
        directions (int): The number of random directions to be generated.
        pow_min_dist (int): 2^pow_min_dist is the smallest distance to be used in the multi-scale analysis.
        pow_max_dist (int): 2^pow_max_dist is the largest distance to be used in the multi-scale analysis.
        scales (int): The number of scales to be used in the multi-scale analysis.
        device (torch.device): The device (CPU or GPU) used for the analysis.
        distances (array): The distances to step along the specified directions.
        directions (list): The normalized directions in the parameter space.
        profiles (array): The loss profiles measured along the specified directions.
        roughness (array): The roughness profiles measured along the specified directions.

    TODO: Consider what parts are best done on what device and move data accordingly.
    TODO: Consider how to parallelize the computation of the loss profiles. JIT? TorchScript?
    TODO: Should I replace numpy arrays with torch tensors for everything that touches the GPU?
    TODO: Decide wether to return the results or store them in the class attributes, or both. Should all methods work the same way?
    TODO: Should I add a class that contains the loss function, so that we can compute the loss all in one go? Would help with jit.fork.
    TODO: Is keeping distances and distances_w_0 redundant ugly? should the methods add the zeros internally? it makes staticmethods easier though?
    TODO: keep the user informed about the progress of the computation. Progress bars, and print.
    TODO: enable the user to manage device? Implement .to() method for cartographer?
    """
    def __init__(
        self,
        model: LossLocus,
        dataset: Dataset,
        criterion: _Loss,
        num_directions: int = 3,
        min_oom: int = -9,
        max_oom: int = 1,
    ) -> None:
        # Turn gradients off
        torch.set_grad_enabled(False)

        # Check if a GPU is available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            warn("No GPU available. Running on CPU.")

        print(f'Initialising Cartographer on {self.device} device')

        # Validate the inputs
        self._validate_inputs(self.device, model, dataset, criterion, num_directions, min_oom, max_oom)
        # create the biggest possible dataloader that fits into the device memory
        self.dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        self.criterion = criterion

        self.descr = f'{model.__class__.__name__} on {dataset.__class__.__name__}'
        print(f'Charting the loss landscape of {self.descr}')

        self.center = LossLocus(model, self.criterion, self.dataloader)
        
        self.DIRECTIONS = num_directions
        # convert min and max oom from powers of 10 to powers of 2
        # log2(10) = 3.32...
        # take the floor of the min and the ceil of the max, so that we definetely include the intended endpoints.
        self.POW_MIN_DIST = int(np.floor(min_oom*np.log2(10)))
        self.POW_MAX_DIST = int(np.ceil(max_oom*np.log2(10)))
        self.SCALES = self.POW_MAX_DIST - self.POW_MIN_DIST + 1 # +1 because we include the endpoints

        print(f'Cartographer initialised with {self.DIRECTIONS} directions and {self.SCALES} sampling distances')

        self.center.to(self.device)

        self.distances = self.generate_distances()
        self.distances.flags.writeable = False
        # an array that includes a row of 0s for the center
        self.distances_w_0 = np.vstack((np.zeros((1, self.DIRECTIONS)), self.distances))
        self.distances_w_0.flags.writeable = False

        self.directions = self.generate_directions()

        # init the profiles array, filled with NaNs
        # longdouble to avoid any precision artefacts.
        self.profiles = np.full((self.SCALES + 1, self.DIRECTIONS), np.nan, dtype=np.longdouble)

    def __call__(self) -> None:
        """
        Generates the locations in parameter space at which the loss
        will be measured, computes the loss profiles and roughness for each scale, and
        stores the results in the class attributes `profiles` and `roughness`.

        Then plots the loss profiles and roughness.
        TODO: Think more about the usecase. Should init already do all this?
        """
        self.measure_profiles()
        self.roughness = self.roughness(self.profiles, self.distances_w_0)

        self.plot()
    
    # TODO: Make this work as intended. This version causes a CUDA runtime error if the dataset + model is too large.
    # @staticmethod
    # def dataloader(device: torch.device, dataset: Dataset) -> DataLoader:
    #     """
    #     See how much of the dataset can be loaded into the device memory.
    #     TODO: what is the best way to proceed if device is CPU? does it matter?
    #     TODO: we do something similar when we load the model to the device, load and catch the exception.
    #     can we combine the try catch into a fixture or something similar?
    #     Returns:
    #         DataLoader: The largest possible DataLoader that can be loaded into the device memory.
    #     """
    #     print(f"Creating DataLoader, trying to load as much of the dataset into the device memory as possible")
    #     # try to load the whole dataset to the device, if possible.
    #     denominator = 1
    #     while True:
    #         try:
    #             dataloader = DataLoader(dataset, batch_size=len(dataset)//denominator, shuffle=False)
    #             data, target = next(iter(dataloader))
    #             data.to(device)
    #             target.to(device)
    #             # if it works, break the loop
    #             break
    #         # if we run out of memory, catch the exception
    #         except RuntimeError as e:
    #             # try with a smaller batch size
    #             denominator += 1
    #     if denominator > 1:
    #         print(f'Split the dataset into {denominator} batches, each of size {len(dataset)//denominator}. Loaded first batch to device.')
    #     else:
    #         print(f'Loaded the whole dataset to device in a single batch')
    #     return dataloader


    def generate_distances(self) -> array:
        """
        Generate how far to step along the various directions.
        Each step is twice as large as the previous.
        Different step sizes for different directions, to avoid artefacts of hitting special frequencies in the model.
        If we assume our model is in float32 the model smallest scale the model can resolve is 1.1754944e-38.
        And the largest parameter in gpt2 is 1e+8.
        Log_2(1e8/1.1754944e-38) = 152.6. That's the number of doublings required to go from the smallest scale to the largest.

        Learning rates are typically in the range of 1e-6 to 1e-2. 
        So there is some reason to belive that that is where the interesting stuff is happening.
        
        Returns:
            array: distances for each direction.
                Dimensions: (num_scales, num_directions)
        """
        print(f'Generating logarithmically spaced distances')
        # To avoid hitting special frequencies in the model, we use different step sizes for different directions.
        # This way we get a logarithmically spaced set of distances.
        dir_steps = 2. ** np.linspace(0, 1, self.DIRECTIONS, endpoint=False)

        # For a given direction, we step from 2^pow_min_dist to 2^pow_max_dist
        # 2^pow_max_dist should be included, so we add 1 to the range
        scale_steps = 2. ** np.arange(self.POW_MIN_DIST, self.POW_MAX_DIST + 1)

        # Compute the matrix of distances using np.outer
        distances = np.outer(scale_steps, dir_steps)
        
        assert distances.shape == (self.SCALES,self.DIRECTIONS)

        return distances
            
    def generate_directions(self) -> [LossLocus]:
        """
        Generates a set of random, normalised directions in the parameter space.
        TODO: Add the option to specify the directions, setting certain parameter groups to zero.
        TODO: Add directions based on gradients.

        Returns:
            [LossLocus]: A list containing the normalised direction vectors in locus objects.
                Dimensions: (num_directions)
        """
        print(f'Generating random directions to explore in parameter space')
        
        # Create a list of random, normalised loci in the parameter space
        directions = [self.center.random_direction() for _ in range(self.DIRECTIONS)]

        # load the models to the device
        for direction in directions:
            direction.to(self.device)
        
        return directions

    @staticmethod
    def _shift(
        point: LossLocus,
        direction: LossLocus,
        distance: float
    ) -> LossLocus:
        """
        Creates a new vector in parameter space.
        returns point + direction * distance
        Static method to make it a pure function.
        Designed for use with torch.jit.fork.

        Args:
            point (LossLocus): The starting point in parameter space.
            direction (LossLocus): The direction in which to move.
            distance (float): The distance to move along the direction.

        Returns:
            LossLocus: The new point in parameter space.
        """
        # validate the inputs
        if not isinstance(point, LossLocus):
            raise TypeError(f"Expected 'point' type LossLocus, got {type(point)}")
        if not isinstance(distance, float):
            raise TypeError(f"Expected 'distance' type float, got {type(distance)}")
        if not isinstance(direction, LossLocus):
            raise TypeError(f"Expected 'direction' type LossLocus, got {type(direction)}")
        # Use not in place multiplication to multiply the direction by the distance, and create a new ParameterVector in the process.
        shift = direction * distance
        # Use in place addition to add the shift to the point, without creating a new object.
        shift += point

        return shift

    def measure_profiles(self) -> None:
        """
        Measures the loss profiles for all distances and directions, as well as the center.
        Uses the _shift to generate the locations in parameter space.
        Uses LossLocus.loss() to compute the loss at each location.

        Writes results directly to class attribute self.profiles.
        """
        # if we have a GPU, use the parallel method, otherwise use the serial method.
        if self.device.type == "cuda":
            self.measure_profiles_parallel()
        else:
            self.measure_profiles_serial()
        
        # check that the profiles array is filled with numbers
        if np.any(np.isnan(self.profiles)):
            raise ValueError("An error occured during the measurement of the loss profiles. Some entries are still np.nan.")

    def measure_profiles_serial(self) -> None:
        """
        Measures the loss profiles for all distances and directions, as well as the center.
        Uses the _shift to generate the locations in parameter space.

        Writes results directly to class attribute self.profiles.
        Don't use this method if you have a GPU available, it will be slow.
        Uses LossLocus.loss() to compute the loss at each location, which goes through the entire dataloader.
        Obviously inefficient.
        TODO: add progress bar.
        """
        warn('Measuring losses serially, because no GPU is available. This will be slow.')

        # loop over all distances and directions
        for i_dist in range(self.SCALES):
            for i_dir in range(self.DIRECTIONS):
                # debug print
                print(f"Measuring loss at distance {self.distances[i_dist][i_dir]} in direction {i_dir}")
                # create a new model by shifting the center in the specified direction by the specified distance
                location = self._shift(self.center, self.directions[i_dir], self.distances[i_dist][i_dir])
                location.to(self.device)
                # compute the loss for the model
                self.profiles[i_dist+1, i_dir] = location.loss()

        # fill the profiles[0][:] with the loss at the center.
        self.center.to(self.device)
        self.profiles[0, :] = self.center.loss()
    
    def measure_profiles_parallel(self) -> None:
        """
        Measures the loss profiles for all distances and directions, as well as the center
        Uses torch.jit.fork to parallelize the computation of the loss profiles.
        Uses the _shift to generate the locations in parameter space.
        Uses the _measure_loss to compute the loss at each location.

        We will try and load as many models as possible into GPU memory as possible.

        Writes results directly to class attribute self.profiles.

        TODO: Add evaluation of the center to the profiles.
        """
        print(f'Measuring losses in parallel, using torch.jit.fork')

        # create a stack of tuples indexing all directions and distances
        # We will iterate over this stack to load models into GPU memory.
        stack = [(i_dist, i_dir) 
                for i_dir in range(self.DIRECTIONS)
                for i_dist in range(self.SCALES)]

        # add an entry for the center, we will catch the -1 later.
        stack.append((-1, 0))

        # Now we will load as many models as possible into GPU memory as possible.
        gpu_batch = {}

        # while the stack is not empty
        while stack:
            try:
                # get the next direction and distance from the stack
                # don't pop it yet, because we might run out of memory, and then we lose track of that entry.
                i_dist, i_dir = stack[-1]
                if i_dist == -1:
                    # if we are at the center, we don't need to shift it.
                    locus = self.center
                else:
                    # create a new model by shifting the center in the specified direction by the specified distance
                    locus = self._shift(self.center, self.directions[i_dir], self.distances[i_dist][i_dir])
                # move the model to the device
                locus.to(self.device) # TODO: is this necessary? or was the model already on the device because it was created from the center?
                # add the model to the gpu_batch
                gpu_batch[(i_dist, i_dir)] = locus
                # if everything works, pop the direction and distance from the stack
                stack.pop()

            # If we run out of memory, we will have to compute the loss for the models we have loaded.
            # or if we have reached the end of the shifts
            except RuntimeError as e:
                self.parallel_evaluate(gpu_batch)
        
        # After the loop, we will have some models left in the locus_batch that we need to evaluate.
        self.parallel_evaluate(gpu_batch)

        # profiles now contains the loss profiles for all directions and distances, and the center at (0,0)
        # fill the profiles[0][:] with the loss at the center.
        self.profiles[0, :] = self.profiles[0, 0]

    def parallel_evaluate(self, locus_batch: {(int,int) : LossLocus}) -> None:
        """
        Evaluates the loss for a batch of models in parallel.
        Uses torch.jit.fork to parallelize the computation of the loss profiles.
        Stores the results in the profiles array.

        Args:
            locus_batch {(int,int): jit.ScriptModule}: A dict of models to evaluate. The keys are tuples of indices (i_dist, i_dir).
        """
        print(f"Evaluating a batch of {len(locus_batch)} jit.ScriptModules in parallel")

        # validation
        if not isinstance(locus_batch, dict):
            raise TypeError(f"Expected 'locus_batch' to be type dict, got {type(locus_batch)}")
        if not all(isinstance(model, LossLocus) for model in locus_batch.values()):
            raise TypeError(f"Expected all values in 'locus_batch' to be of type LossLocus")
        if not all(isinstance(model.loss_script, jit.ScriptModule) for model in locus_batch.values()):
            raise TypeError(f"Expected all values in 'locus_batch' to be of type jit.ScriptModule")
        if not all(isinstance(indices, tuple) and len(indices) == 2 for indices in locus_batch.keys()):
            raise TypeError(f"Expected all keys in 'locus_batch' to be tuples of length 2")

        # Out of bounds check
        for i_dist, i_dir in locus_batch.keys():
            if not i_dist in range(-1,self.SCALES):
                raise ValueError(f"Expected -1 <= i_dist <= {self.SCALES-1}, got i_dist={i_dist}.")
            if not i_dir in range(self.DIRECTIONS):
                raise ValueError(f"Expected 0 <= i_dir <= {self.DIRECTIONS-1}, got i_dir={i_dir}.")

        # TODO: is this the best place to do this? redundant?
        for locus in locus_batch.values():
            locus.to(self.device)

        # set the values of the profiles array to zero
        # so that we can accumulate the loss values in them.
        for indices in locus_batch.keys():
            i_dist, i_dir = indices
            if not np.isnan(self.profiles[i_dist+1, i_dir]):
                raise ValueError(f"Expected profiles[{i_dist+1}, {i_dir}] to be np.isnan(), got {self.profiles[i_dist+1, i_dir]}")
            self.profiles[i_dist+1, i_dir] = 0.0

        # loop over dataloader
        for data, target in self.dataloader:
            # move the data and target to the device
            data = data.to(self.device)
            target = target.to(self.device)
            # compute the loss for the models we have loaded with jit.fork
            # create a dict of futures
            futures = {indices: torch.jit.fork(locus.loss_script, data, target)
                        for indices, locus in locus_batch.items()}
            # wait for the futures to complete
            for indices, future in futures.items():
                i_dist, i_dir = indices
                loss = np.longdouble(future.wait().item())
                self.profiles[i_dist+1, i_dir] += loss/len(self.dataloader) # the +1 is because the first entry is the center.

        # free GPU memory by clearing the locus_batch
        locus_batch.clear()

        # tell cuda to free memory
        torch.cuda.empty_cache()

    @staticmethod
    def roughness(losses: array, distances_w_0: array) -> array:
        """
        TODO: Maybe it should be called 'grit' or 'grit size' instead of 'roughness'? Like sandpaper grit.
        or is the grit size of a landscape the scale at which the roughness is largest? no that's called particle size in sandpaper.
        Measures the roughness for all triples of points on the loss profile.
        Consider you measure the following points: 
        A = (a,loss(a))
        B = (b,loss(b))
        C = (c,loss(c))

        a,b and c are spaced equally in parameter space:
            
           L|                B
           O|                |           C
           S|    A           |           |
           S|____|___________|___________|___
                    Parameter Space
        
        We can now define "roughness" at the scale of dist(a,b) by measuring how much B deviates from the straight line between A and C.
        The metric we will use is inspired by the coastline paradox:
        We will measure the length of the curve formed by the path ABC and devide it by the straight line distance from A to C.
        This ratio is what we will call the roughness at the scale of ~dist(a,b).

        Our distance array was generated such, that the distance between a point (b) and the center (a)
        is exactly equal to the distance between the point (c) and its successor in the distance array (b).

        Note of interest: Take an ellipse with A and C as the foci and B as a point on the ellipse.
        Then roughness is the ratio of the length of semi-major axis to the linear eccentricity.
        Whonder what this means. It's definitely useful for inverting roughness to recover B.

        Args:
            distances_w_0 (array): The distance of each point from the center, including the center. (ie. a row of 0s)
                Dimensions: (num_scales+1, num_directions)
            losses (array): The loss profiles measured along the specified directions.
                Dimensions: (num_scales+1, num_directions)

        Returns:
            array: roughness at each scale, in each direction.
                Dimensions: (num_scales-1, num_directions), since no roughness can be calculated for the start and end points.
        """

        print(f'Calculating roughness')

        # Validation
        if not isinstance(distances_w_0, np.ndarray):
            raise TypeError(f"Expected 'distances_w_0' type np.ndarray, got {type(distances_w_0)}")
        if not isinstance(losses, np.ndarray):
            raise TypeError(f"Expected 'profiles' type np.ndarray, got {type(losses)}")
        # Test that the distances are positive
        if not np.all(distances_w_0 >= 0):
            raise ValueError(f"Expected all distances to be positive, got {distances_w_0}")
        # check that the first entry in the profiles array (center) is the same for all directions
        if not np.all(losses[0, :] == losses[0, 0]):
            raise ValueError(f"Expected all profiles[0, :] to be the same because they represent the center, got {losses[0, :]}")

        # prepend 0 to the distances array, to represent the distance from the center to the center.
        # rename to distA to underscore that it is the distance as measured from A
        distA = distances_w_0

        # check that the shapes are now correct
        if not distA.shape == losses.shape:
            raise ValueError(f"Expected distances and losses to have the same shape after accounting for the center, got {distA.shape} and {losses.shape}")

        # Define slices for the triples
        a = 0               # a is always the center
        b = slice(1, -1)    # b is in the middle, but never the first or last entry of dist/losses so exclude one on either side
        c = slice(2, None)  # c is the last of the triplet, and must always have to predecessors (a and b) so exclude the first two

        # Perform calculations directly using slices
        distB_c = distA[b] # = distA[c] - distA[b], because distA[c] = 2*distA[b]

        # use this to verify that the distances are correct
        if not np.all(distA[c] == 2*distA[b]):
            raise ValueError(f"Expected distA[c] = 2*distA[b], got {distA[c]} and {2*distA[b]}")

        AB = np.sqrt((losses[b] - losses[a])**2 + distA[b]**2) # the distance from the center of
        BC = np.sqrt((losses[c] - losses[b])**2 + distB_c**2)
        AC = np.sqrt((losses[c] - losses[a])**2 + distA[c]**2) 

        # Calculate roughness
        roughness = (AB + BC) / AC

        return roughness
    
    # @staticmethod
    # def map(profiles: array, distances_w_0: array) -> plt.Figure:
    #     pass

    def plot(self) -> None:
        """
        Create and display all plots
        """
        print(f'Plotting results')

        # convert everything to double precision because plotly doesn't work with longdouble.
        profiles64 = np.double(self.profiles)
        distances_w_0_64 = np.double(self.distances_w_0)
        roughness64 = np.double(self.roughness)
        distances64 = np.double(self.distances)

        # Plot the loss profiles
        fig = self.plot_profiles(profiles64, distances_w_0_64)
        fig.layout.title = f'Loss Landscape of {self.descr}'
        fig.show()

        # Plot the roughness
        fig = self.plot_roughness(roughness64, distances64)
        fig.layout.title = f'Scale Dependent Roughness of {self.descr}'
        fig.show()

        
    @staticmethod
    def plot_profiles(losses: array, distances_w_0: array) -> go.Figure:
        """
        Plot the loss sampled at different distances from the center along different directions.

        Args:
            profiles (array): The loss profiles measured along the some directions.
                Dimensions: (num_scales+1, num_directions)
            distances (array): INCLUDING 0! The distance of each point from the center, including the center. (ie. a row of 0s)
                Dimensions: (num_scales+1, num_directions)

        Returns:
            go.Figure: The figure containing the loss.

        TODO: clean up the code.
        TODO: add typical learning rate ranges to the plot as vertical lines.
        TODO: add minimum precision of float32 for loss(center) to the plot as two horizontal lines above and below the center loss
        
        """
        
        # validate the input
        if not isinstance(losses, np.ndarray) or not isinstance(distances_w_0, np.ndarray):
            raise ValueError(f'Both losses and distances_w_0 must be numpy arrays, but got {type(losses)} and {type(distances_w_0)} respectively')
        if not losses.shape == distances_w_0.shape:
            raise ValueError(f'Both losses and distances_w_0 must have the same shape, but got {losses.shape} and {distances.shape} respectively')
        if not all(distances_w_0[0] == 0):
            raise ValueError(f'The first column of distances_w_0 must be zero, but got {distances_w_0[0]} instead')
        
        _, NR_DIRS = losses.shape

        # copy the distances array, so that we can modify it
        distances = distances_w_0.copy()

        flip = True  # Whether to flip half the directions or not

        if flip:
            distances[:, 1::2] = -distances[:, 1::2]  # Negate every second direction

        fig = go.Figure()

        # Adding traces
        for i in range(NR_DIRS):
            #ic(distances[:, i], losses[:, i])
            #ic(len(distances[:, i]), len(losses[:, i]))
            fig.add_trace(go.Scatter(
                x=distances[:, i], 
                y=losses[:, i], 
                mode='lines+markers',
                name=f'Direction {i+1}',
                #marker = dict(symbol='cross')
            ))

        # Center the axes around the [center, loss(center)], so that we can start zooming out of the box
        # Calculate maximum distance for symmetric x-axis range
        max_distance = np.max(np.abs(distances))

        # y axis range: center on loss(center) and extend to the maximum loss
        center_loss = losses[0, 0]
        max_diff = np.max(np.abs(losses-center_loss))

        fig.update_layout(
            title=f'Loss Landscape',
            xaxis = dict(
                title='Distance from Center',
                range=[-max_distance, max_distance],
                tickformat='.0e',
            ),
            yaxis = dict(
                title='Loss',
                range=[center_loss - max_diff, center_loss + max_diff],
            ),
            dragmode='zoom',
            hovermode='closest'
        )

        return fig

    @staticmethod
    def plot_roughness(roughness: array, distances: array) -> go.Figure:
        """
        Plots the roughness at different 'COARSE GRAIN SCALES' for multiple directions, as contained in the roughness array.
        
        Args:
            roughness (np.ndarray): The roughness array.
                Dimensions: (num_scales-1, num_directions)
            distances (np.ndarray): Distances without 0.
                Dimensions: (num_scales, num_directions)

        Returns:
            go.Figure: The figure containing the roughness profiles plot.
        """
        # validate the input
        if not isinstance(roughness, np.ndarray):
            raise ValueError(f"The roughness must be a numpy array, but it is {type(roughness)}.")
        if not isinstance(distances, np.ndarray):
            raise ValueError(f"The distances must be a numpy array, but it is {type(distances)}.")

        _, NR_DIRS = roughness.shape

        # copy distances to avoid modifying the original array
        distances = distances.copy()

        # remove the longest distance, because there is no roughness value for it
        distances = distances[:-1]

        if not roughness.shape == distances.shape:
            raise ValueError(f"The roughness and distances[:-1] must have the same shape, but they are {roughness.shape} and {distances.shape}.")
        
        # create the figure
        fig = go.Figure()

        # iterate over all directions and plot them
        for i in range(NR_DIRS):
            fig.add_trace(go.Scatter(
                x=distances[:, i],
                y=roughness[:, i],
                mode='markers',
                name=f'Direction {i}'
            ))
        
        fig.update_layout(
            title=f'Scale Dependent Roughness',
            xaxis_title='Coarse Graining Scale',
            xaxis=dict(
                type='log',
                exponentformat='power',
            ),
            yaxis_title='Roughness',
            dragmode='zoom',
            hovermode='closest'
        )

        return fig

    def _validate_inputs(
        self,
        device: torch.device,
        model: nn.Module,
        dataset: Dataset,
        criterion: _Loss,
        directions: int,
        pow_min_dist: int,
        pow_max_dist: int,
    ) -> None:
        """
        Validates the input arguments for the Cartographer class.

        Raises:
            ValueError: If any of the input arguments are invalid.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("'model' must be an instance of torch.nn.Module")

        if not isinstance(dataset, Dataset):
            raise TypeError("'dataset' must be an instance of torch.utils.data.Dataset")

        if not isinstance(criterion, _Loss):
            raise TypeError("'criterion must be an instance of torch.nn.modules.loss._Loss")

        if not isinstance(directions, int) or directions < 0:
            raise ValueError("'directions' must be a positive integer")

        if not isinstance(pow_min_dist, int):
            raise TypeError("'pow_min_dist' must be an integer.")

        if not isinstance(pow_max_dist, int):
            raise TypeError("'pow_max_dist' must be an integer.")
        
        if not pow_min_dist <= pow_max_dist:
            raise ValueError("'pow_min_dist' must be smaller equal 'pow_max_dist'")

        # Check for operational readiness of model and dataloader
        try:
            model.eval()  # Ensure model can be set to evaluation mode
        except AttributeError as e:
            raise ValueError(f"Model does not support evaluation mode: {str(e)}")

        # Test try making a dataloader from the dataset
        try:
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            sample_input, sample_target = next(iter(dataloader))
        except TypeError as e:
            raise ValueError(f"Dataset does not produce an iterable dataloader or does not produce outputs and targets: {str(e)}")

        # try loading everything to device
        try:
            model.to(device)
            criterion.to(device)
            sample_input = sample_input.to(device)
            sample_target= sample_target.to(device)
        except Exception as e:
            raise ValueError(f"Failed to load model, criterion, input or target to device: {str(e)}")

        # Model output compatibility with loss function
        try:
            test_output = model(sample_input)
        except Exception as e:
            raise ValueError(f"Model failed to process input from DataLoader: {str(e)}")
        
        # Target and output compatibility with loss function
        try:
            criterion(test_output, sample_target)
        except Exception as e:
            raise ValueError(f"Loss function cannot process model output: {str(e)}")