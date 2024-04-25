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
"""

import torch
from torch import nn
from torch import jit
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.nn import ModuleList, ModuleDict
from parameter_vector import LossLocus # vector operations on nn.Module
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import array

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
        center (ParameterVector): The machine learning model to be analyzed.
        dataloader (DataLoader): The dataset used to compute the loss. For deterministic loss we iterate over the entire dataset. Make sure to use a small dataset.
        loss_function (_Loss): The loss function used to evaluate the model.
        directions (int): The number of random directions to be generated.
        pow_min_dist (int): 2^pow_min_dist is the smallest distance to be used in the multi-scale analysis.
        pow_max_dist (int): 2^pow_max_dist is the largest distance to be used in the multi-scale analysis.
        scales (int): The number of scales to be used in the multi-scale analysis.
        device (torch.device): The device (CPU or GPU) used for the analysis.
        distances (array): The distances to step along the specified directions.
        directions (ModuleList): The normalized directions in the parameter space.
        profiles (array): The loss profiles measured along the specified directions.
        roughness (array): The roughness profiles measured along the specified directions.

    TODO: Make sure the model is always in evaluation mode no_grad().
    TODO: Consider what parts are best done on what device and move data accordingly.
    TODO: Consider how to parallelize the computation of the loss profiles. JIT? TorchScript?
    TODO: Should I replace numpy arrays with torch tensors for everything that touches the GPU?
    TODO: Decide wether to return the results or store them in the class attributes, or both. Should all methods work the same way?
    TODO: Should I add a class that contains the loss function, so that we can compute the loss all in one go? Would help with jit.fork.
    """
    def __init__(
        self,
        model: LossLocus,
        dataloader: DataLoader,
        criterion: _Loss,
        num_directions: int = 3,
        pow_min_dist: int = -21,
        pow_max_dist: int = 3,
    ) -> None:
        self._validate_inputs(model, dataloader, criterion, num_directions, pow_min_dist, pow_max_dist)
        self.model_class = model.__class__
        self.center = model
        self.dataloader = dataloader
        self.loss_function = criterion
        self.num_directions = num_directions
        self.pow_min_dist = pow_min_dist
        self.pow_max_dist = pow_max_dist
        self.num_scales = pow_max_dist - pow_min_dist + 1 # +1 because we include the endpoints

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.center.to(self.device)
        self.dataloader.to(self.device)
        # Turn gradients off
        torch.set_grad_enabled(False)
        # set the model to evaluation mode
        self.center.eval()

        self.distances = self.generate_distances()
        self.directions = self.generate_directions()

        # init the profiles and roughness arrays, filled with NaNs
        self.profiles = np.full((self.num_scales + 1, self.num_directions), float('nan'))
        self.roughness = np.full((self.num_scales - 1, self.num_directions), float('nan'))

    def __call__(self) -> None:
        """
        Generates the locations in parameter space at which the loss
        will be measured, computes the loss profiles and roughness for each scale, and
        stores the results in the class attributes `profiles` and `roughness`.
        """
        self.locations = self.generate_locations()
        self.profiles = self.measure_loss()
        self.roughness = self.measure_roughness()

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

        # To avoid hitting special frequencies in the model, we use different step sizes for different directions.
        # This way we get a logarithmically spaced set of distances.
        dir_steps = 2. ** np.linspace(0, 1, self.num_directions, endpoint=False)

        # For a given direction, we step from 2^pow_min_dist to 2^pow_max_dist
        # 2^pow_max_dist should be included, so we add 1 to the range
        scale_steps = 2. ** np.arange(self.pow_min_dist, self.pow_max_dist + 1)

        # Compute the matrix of distances using np.outer
        distances = np.outer(scale_steps, dir_steps)
        
        assert distances.shape == (self.num_scales,self.num_directions)

        return distances
            
    def generate_directions(self) -> ModuleList:
        """
        Generates a set of random, normalised directions in the parameter space.
        TODO: Add the option to specify the directions, setting certain parameter groups to zero.

        Returns:
            ModuleList: A ModuleList containing the nomralised direction vectors.
            These vectors are model parameters.
                Dimensions: (num_directions)
        """
        # Create a ModuleList of length num_directions, with an instance of model_class for each entry
        directions = ModuleList([self.model_class() for _ in range(self.num_directions)])
        
        # Normalize the directions
        for direction in directions:
            direction /= abs(direction)
        
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
            point (ParameterVector): The starting point in parameter space.
            direction (ParameterVector): The direction in which to move.
            distance (float): The distance to move along the direction.

        Returns:
            ParameterVector: The new point in parameter space.
        """
        # validate the inputs
        if not isinstance(point, LossLocus):
            raise TypeError(f"Expected 'point' type ParameterVector, got {type(point)}")
        if not isinstance(distance, float):
            raise TypeError(f"Expected 'distance' type float, got {type(distance)}")
        if not isinstance(direction, LossLocus):
            raise TypeError(f"Expected 'direction' type ParameterVector, got {type(direction)}")
        # Use not in place multiplication to multiply the direction by the distance, and create a new ParameterVector in the process.
        shift = direction * distance
        # Use in place addition to add the shift to the point, without creating a new object.
        shift += point

        return shift

    def measure_profiles(self) -> array:
        """
        Measures the loss profiles for all distances and directions, as well as the center
        Uses torch.jit.fork to parallelize the computation of the loss profiles.
        Uses the _shift to generate the locations in parameter space.
        Uses the _measure_loss to compute the loss at each location.

        We will try and load as many models as possible into GPU memory as possible.

        Writes results directly to class attribute self.profiles.

        TODO: Add evaluation of the center to the profiles.

        Returns:
            array: loss profiles at each scale, in each direction.
                Dimensions: (num_scales+1, num_directions)
        """

        # move the models to the device
        self.center.to(self.device)
        self.directions.to(self.device)

        # create a stack of tuples indexing all directions and distances
        # We will iterate over this stack to load models into GPU memory.
        stack = [(i_dist, i_dir) for i_dir in range(num_directions) for i_dist in range(num_scales)]

        # add an entry for the center, 
        stack.append((-1, 0))

        # Now we will load as many models as possible into GPU memory as possible.
        gpu_batch = ModuleDict()

        # while the stack is not empty
        while stack:
            try:
                # get the next direction and distance from the stack
                # don't pop it yet, because we might run out of memory, and then we lose track of that entry.
                i_dist, i_dir = stack[-1]
                if i_dist == -1:
                    # if we are at the center, we don't need to shift it.
                    model = self.center
                else:
                    # create a new model by shifting the center in the specified direction by the specified distance
                    model = self._shift(self.center, self.directions[i_dir], self.distances[i_dist][i_dir])
                # move the model to the device
                model.to(self.device) # TODO: is this necessary? or was the model already on the device because it was created from the center?
                # add the model to the gpu_batch
                gpu_batch[(i_dist, i_dir)] = model
                # if this pop the direction and distance from the stack
                stack.pop()

            # If we run out of memory, we will have to compute the loss for the models we have loaded.
            # or if we have reached the end of the shifts
            except RuntimeError as e:
                self.parallel_evaluate(gpu_batch)
        
        # After the loop, we will have some models left in the model_batch that we need to evaluate.
        self.parallel_evaluate(gpu_batch)

        # profiles now contains the loss profiles for all directions and distances, and the center at (0,0)
        # fill the profiles[0][:] with the loss at the center.
        self.profiles[0, :] = self.profiles[0, 0]

        # normalize the profiles, by dividing by the number of samples in the dataloader.
        self.profiles /= len(self.dataloader)

        return self.profiles

    @staticmethod
    def parallel_evaluate(model_batch: ModuleDict(), dataloader: DataLoader, results: array) -> array:
        """
        Evaluates the loss for a batch of models in parallel.
        Uses torch.jit.fork to parallelize the computation of the loss profiles.
        Uses the _measure_loss to compute the loss at each location.

        Args:
            model_batch (ModuleDict): {(int,int): nn.Module}
                                    A dict of models to evaluate. The keys are tuples of indices (i_dist, i_dir).
            dataloader (DataLoader): The dataset to evaluate the models on.
            results (array): The array to store the results in. The loss is stored at result[i_dist+1, i_dir].

        Returns:
            array: The results array, filled with the losses.
        """

        # validation
        if not isinstance(model_batch, ModuleDict):
            raise TypeError(f"Expected 'model_batch' type ModuleDict, got {type(model_batch)}")
        if not all(isinstance(model, nn.Module) for model in model_batch.values()):
            raise TypeError(f"Expected all values in 'model_batch' to be of type nn.Module")
        if not all(isinstance(indices, tuple) and len(indices) == 2 for indices in model_batch.keys()):
            raise TypeError(f"Expected all keys in 'model_batch' to be tuples of length 2")
        if not isinstance(results, array):
            raise TypeError(f"Expected 'results' type array, got {type(results)}")
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f"Expected 'dataloader' type DataLoader, got {type(dataloader)}")

        # Out of bounds check
        for i_dist, i_dir in model_batch.keys():
            if not (0 <= i_dist + 1 < num_dists and 0 <= i_dir < num_dirs):
                raise IndexError(f"Indices ({i_dist+1}, {i_dir}) are out of bounds for the results array")
        
        # loop over dataloader
        for data, target in dataloader:
            # compute the loss for the models we have loaded with jit.fork
            # create a dict of futures
            future_outputs = {indices: torch.jit.fork(model, data)
                        for indices, model in model_batch.items()}
            # feed the results to the loss function
            # maybe sketchy: feeding wait() results directly into a new fork
            # future_losses = {indices: torch.jit.fork(self.loss_function, future_output.wait(), target)
            #             for indices, future_output in future_outputs.items()}
            # simpler version that doesn't use jit.fork.
            # TODO: test if the above version is faster and works. Or make a module that contains the loss function, then we only need one fork.
            future_losses = {indices: self.loss_function(future_output.wait(), target)
                        for indices, future_output in future_outputs.items()}

            # wait for the futures to complete
            for indeces, future_loss in future_losses.items():
                i_dist, i_dir = indeces
                loss = future_loss #.wait()
                self.profiles[i_dist+1, i_dir] += loss.item() # the +1 is because the first entry is the center.

        # clear the model_batch, TODO: does this clear the GPU memory?
        model_batch.clear()
        model_batch = ModuleDict()

    # @jit.script
    # def _measure_loss(
    #     model,
    #     dataloader: DataLoader,
    #     loss_function: _Loss,
    # ) -> float:
    #     """
    #     Measures the loss of a model on a dataset using a given loss function.
    #     Static method to make it a pure function.
    #     Designed for use with torch.jit.fork.

    #     Warning:Iterates over the entire dataset to ensure deterministic loss.
    #     Choose an appropriately small dataset to avoid long computation times.
        
    #     Args:
    #         model (nn.Module): The model to evaluate.
    #         dataloader (Da    # @jit.script
    # def _measure_loss(
    #     model,
    #     dataloader: DataLoader,
    #     loss_function: _Loss,
    # ) -> float:
    #     """
    #     Measures the loss of a model on a dataset using a given loss function.
    #     Static method to make it a pure function.
    #     Designed for use with torch.jit.fork.

    #     Warning:Iterates over the entire dataset to ensure deterministic loss.
    #     Choose an appropriately small dataset to avoid long computation times.
        
    #     Args:
    #         model (nn.Module): The model to evaluate.
    #         dataloader (DataLoader): The dataset to evaluate the model on.
    #         loss_function (_Loss): The loss function to use.

    #     Returns:
    #         float: The loss of the model on the dataset.
    #     """
    #     # Validate the inputs
    #     if not isinstance(model, ParameterVector):
    #         raise TypeError(f"Expected 'model' type ParameterVector, got {type(model)}")
    #     if not isinstance(dataloader, DataLoader):
    #         raise TypeError(f"Expected 'dataloader' type DataLoader, got {type(dataloader)}")
    #     if not isinstance(loss_function, _Loss):
    #         raise TypeError(f"Expected 'loss_function' type _Loss, got {type(loss_function)}")

    #     # Set the model to evaluation mode and disable gradient tracking
    #     # Initialize the total loss to zero
    #     total_loss = 0.0
    #     # Iterate over the dataset
    #     for data, target in dataloader:
    #         # Forward pass
    #         output = model(data)
    #         # Compute the loss
    #         loss = loss_function(output, target)
    #         # Add the loss to the total
    #         total_loss += loss.item()
        
    #     return total_loss
    
    #     # Validate the inputs
    #     if not isinstance(model, ParameterVector):
    #         raise TypeError(f"Expected 'model' type ParameterVector, got {type(model)}")
    #     if not isinstance(dataloader, DataLoader):
    #         raise TypeError(f"Expected 'dataloader' type DataLoader, got {type(dataloader)}")
    #     if not isinstance(loss_function, _Loss):
    #         raise TypeError(f"Expected 'loss_function' type _Loss, got {type(loss_function)}")

    #     # Set the model to evaluation mode and disable gradient tracking
    #     # Initialize the total loss to zero
    #     total_loss = 0.0
    #     # Iterate over the dataset
    #     for data, target in dataloader:
    #         # Forward pass
    #         output = model(data)
    #         # Compute the loss
    #         loss = loss_function(output, target)
    #         # Add the loss to the total
    #         total_loss += loss.item()
        
    #     return total_loss
    

    def measure_roughness(self) -> array:
        """
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

        Since the predecessor of every point is at half the distance to the center, we can calculate the roughness for all triples of points in parallel.

        Returns:
            array: roughness at each scale, in each direction.
                Dimensions: (num_scales-1, num_directions), since no roughness can be calculated for the start and end points.
        """
        pass

    def plot(self) -> plt.Figure:
        """
        Plot the loss landscape and the roughness profiles.

        Returns:
            plt.Figure: The figure containing the loss and roughness plots.
        """
        pass

    def plot_loss(self) -> plt.Figure:
        """
        Plots the loss landscape given by the loss profiles array.
        
        Returns:
            plt.Figure: The figure containing the loss landscape plot.
        """
        pass

    def plot_roughness(self) -> plt.Figure:
        """
        Plots the roughness at different scales for multiple directions, as contained in the roughness array.
        
        Returns:
            plt.Figure: The figure containing the roughness profiles plot.
        """
        pass

    def _validate_inputs(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_function: _Loss,
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

        if not isinstance(dataloader, DataLoader):
            raise TypeError("'dataloader' must be an instance of torch.utils.data.DataLoader")

        if not isinstance(loss_function, _Loss):
            raise TypeError("'loss_function' must be an instance of torch.nn.modules.loss._Loss")

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

        # Test DataLoader output
        try:
            sample_input, sample_target = next(iter(dataloader))
        except TypeError as e:
            raise ValueError(f"DataLoader is not iterable or does not produce outputs and targets: {str(e)}")

        # Model output compatibility with loss function
        try:
            test_output = model(sample_input)
        except Exception as e:
            raise ValueError(f"Model failed to process input from DataLoader: {str(e)}")
        
        # Target and output compatibility with loss function
        try:
            loss_function(test_output, sample_target)
        except Exception as e:
            raise ValueError(f"Loss function cannot process model output: {str(e)}")