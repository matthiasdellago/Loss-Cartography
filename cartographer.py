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

from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from pandas import DataFrame
from parameter_vector import ParameterVector # vector operations on nn.Module
import seaborn as sns
import matplotlib.pyplot as plt

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
        center (nn.Module): The machine learning model to be analyzed.
        dataloader (DataLoader): The dataset used to compute the loss.
        loss_function (_Loss): The loss function used to evaluate the model.
        directions (int): The number of random directions to be generated.
        scales (int): The number of scales to be used in the multi-scale analysis.
        device (torch.device): The device (CPU or GPU) used for the analysis.
        profiles (DataFrame): The loss profiles measured along the specified directions.
        roughness (DataFrame): The roughness profiles measured along the specified directions.
    """
    def __init__(
        self,
        model: Module,
        dataloader: DataLoader,
        loss_function: _Loss,
        directions: int = 3,
        scales: int = 3,
    ) -> None:
        self._validate_inputs(model, dataloader, loss_function, directions, scales)
        self.center = model
        self.dataloader = dataloader
        self.loss_function = loss_function
        self.directions = directions
        self.scales = scales

        # Determine the available device and move everything to it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.center.to(self.device)
        self.loss_function.to(self.device)

        self.distances = self.generate_distances()
        self.directions = self.generate_directions()

    def __call__(self) -> None:
        """
        Generates the locations in parameter space at which the loss
        will be measured, computes the loss profiles and roughness for each scale, and
        stores the results in the class attributes `profiles` and `roughness`.
        """
        self.locations = self.generate_locations()
        self.profiles = self.measure_loss()
        self.roughness = self.measure_roughness()

    def generate_distances(self) -> DataFrame:
        """
        Generate how far to step along the various directions.
        Each step is twice as large as the previous.
        Different step sizes for different directions, to avoid artefacts of hitting special frequencies in the model.

        Returns:
            Dataframe: distances for each direction.
                Dimensions: (scales, directions)
        """
        pass
    
    def generate_directions(self) -> DataFrame:
        """
        Generates a set of random, normalised directions in the parameter space.
        TODO: Add the option to specify the directions, setting certain parameter groups to zero.

        Returns:
            DataFrame: A DataFrame containing the nomralised direction vectors.
            These vectors are model parameters.
                Dimensions: (directions)
        """
        pass
    
    def generate_locations(self) -> DataFrame:
        """
        Generates the locations in parameter space at which the loss will be measured, based on the directions.
        The locations are generated by stepping along the directions, with the step size determined by the scale.

        Returns:
            Dataframe: locations where the loss will be measured.
                Dimensions: (scales, directions)
        """
        pass

    def measure_loss(self) -> DataFrame:
        """
        Measures the loss at all location, including the center.
        Evaluates in parallel on devide using torch.jit.fork.

        Returns:
            Dataframe: loss profiles measured in each direction, starting from the center.
                Dimensions: (scales+1, directions), because we include the center.
        """
        pass

    def measure_roughness(self) -> DataFrame:
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
            Dataframe: roughness at each scale, in each direction.
                Dimensions: (scales-1, directions), since no roughness can be calculated for the start and end points.
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
        Plots the loss landscape given by the loss profiles dataframe.
        
        Returns:
            plt.Figure: The figure containing the loss landscape plot.
        """
        pass

    def plot_roughness(self) -> plt.Figure:
        """
        Plots the roughness at different scales for multiple directions, as contained in the roughness dataframe.
        
        Returns:
            plt.Figure: The figure containing the roughness profiles plot.
        """
        pass

    def _validate_inputs(
        self,
        model: Module,
        dataloader: DataLoader,
        loss_function: _Loss,
        directions: int,
        scales: int,
    ) -> None:
        """
        Validates the input arguments for the Cartographer class.

        Raises:
            ValueError: If any of the input arguments are invalid.
        """
        if not isinstance(model, Module):
            raise ValueError("'model' must be an instance of torch.nn.Module")

        if not isinstance(dataloader, DataLoader):
            raise ValueError("'dataloader' must be an instance of torch.utils.data.DataLoader")

        if not isinstance(loss_function, _Loss):
            raise ValueError("'loss_function' must be an instance of torch.nn.modules.loss._Loss")

        if not isinstance(directions, int) or directions < 0:
            raise ValueError("'directions' must be a positive integer")

        if not isinstance(scales, int) or scales < 0:
            raise ValueError("'scales' must be a positive integer")
