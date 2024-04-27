#test_cartographer.py
import pytest
import torch
from torch.utils.data import Dataset
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from loss_locus import LossLocus
from cartographer import Cartographer
from simple_cnn import SimpleCNN
import numpy as np

# TODO: Find out why pytests run so extremely slow. Running these tests takes 768 seconds on my machine!
# Invoking some of the tests manually takes a few seconds.

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def model(device):
    return SimpleCNN().to(device)

@pytest.fixture
def dataset():
    # Use a standard MNIST normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    return dataset

@pytest.fixture
def criterion():
    return torch.nn.CrossEntropyLoss()

@pytest.fixture
def cartographer(model, dataset, criterion):
    """A fixture that returns a small Cartographer object with 2 directions and a distance range of -1 to 1."""
    return Cartographer(model=model, dataset=dataset, criterion=criterion, num_directions=2, pow_min_dist=-1, pow_max_dist=1)

def test_validate_inputs(model, dataset, criterion):
    # Test validation during __init__ 
    # Test that the Cartographer class validates the input arguments correctly
    with pytest.raises(TypeError):
        Cartographer(model=None, dataset=dataset, criterion=criterion)
    with pytest.raises(TypeError):
        Cartographer(model=model, dataset=None, criterion=criterion)
    with pytest.raises(TypeError):
        Cartographer(model=model, dataset=dataset, criterion=None)
    with pytest.raises(ValueError):
        Cartographer(model=model, dataset=dataset, criterion=criterion, num_directions=-1)
    with pytest.raises(ValueError):
        Cartographer(model=model, dataset=dataset, criterion=criterion, pow_min_dist=3, pow_max_dist=1)
    
    # TODO: These validation tests are not excellent. Improve or remove them.

    # Test model and dataset compatibility
    with pytest.raises(ValueError):
        class IncompatibleModel(nn.Module):
            def forward(self, x):
                raise ValueError("Invalid input")
        incompatible_model = IncompatibleModel()
        Cartographer(model=incompatible_model, dataset=dataset, criterion=criterion)

    # Test target and output compatibility with loss function
    with pytest.raises(ValueError, match="Loss function cannot process model output"):
        class OutputMismatchModel(nn.Module):
            def forward(self, x):
                return x + 1.  # Assuming non-compatible output
        mismatch_model = OutputMismatchModel()
        Cartographer(model=mismatch_model, dataset=dataset, criterion=criterion)


    # Test that the Cartographer class accepts the input arguments correctly
    cartographer = Cartographer(model=model, dataset=dataset, criterion=criterion)
    assert cartographer.model_class is model.__class__
    assert cartographer.criterion is criterion

def test_distance_generation_1(model, dataset, criterion):
    # Test that the distance generation works as expected
    # If num_directions is 1 and pow_min_dist and pow_max_dist are 0, the distances should be of shape (1, 1) and contain only 1.0
    cartographer = Cartographer(model=model, dataset=dataset, criterion=criterion, num_directions=1, pow_min_dist=0, pow_max_dist=0)
    distances = cartographer.generate_distances()

    assert isinstance(distances, np.ndarray), f"Expected type numpy.ndarray, but got {type(distances)}"
    assert distances.shape == (1, 1), f"Unexpected shape: {distances.shape}, expected (1, 1)"
    assert np.isclose(distances[0,0], 1.0), f"Unexpected value at (0,0): {distances[0,0]}, expected 1.0"

def test_distance_generation_manual(model, dataset, criterion):
    # Test that the distance generation works as expected
    # Caclculate distances by hand for num_directions=2, pow_min_dist=0 and pow_max_dist = 2
    cartographer = Cartographer(model=model, dataset=dataset, criterion=criterion, num_directions=2, pow_min_dist=0, pow_max_dist=2)
    distances = cartographer.generate_distances()

    # Hand calculated distances
    manual_distances = np.array([
        [1.00, 1.41421356],
        [2.00, 2.82842712],
        [4.00, 5.65685425],
    ])

    assert isinstance(distances, np.ndarray), f"Expected type numpy.ndarray, but got {type(distances)}"
    assert distances.shape == (3, 2), f"Unexpected shape: {distances.shape}, expected (3, 2)"
    assert np.allclose(distances, manual_distances), f"Unexpected values:\n{distances}\nExpected:\n{manual_distances}"

def test_direction_generation(model, dataset, criterion):
    num = 7
    cartographer = Cartographer(model=model, dataset=dataset, criterion=criterion, num_directions=num)
    directions = cartographer.generate_directions()
    
    assert isinstance(directions, list), f"Expected type nn.ModuleList, but got {type(directions)}"
    # check that all objects in the list are LossLoci
    assert all(isinstance(direction, LossLocus) for direction in directions), f"Expected all directions to be of type {type(model)}, but got {set(type(direction) for direction in directions)}"
    # check that the number of directions is as expected
    assert len(directions) == num, f"Expected {num} directions, but got {len(directions)}"
    # check that the directions are normalized
    for direction in directions:
        assert np.isclose(abs(direction),1.0), f"Expected norm to be 1, but got {abs(direction)}"
    # in a high dimensional space, randomly chosen directions should be quasi-orthogonal
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            params_i = torch.cat([p.data.view(-1) for p in directions[i].parameters()])
            params_j = torch.cat([p.data.view(-1) for p in directions[j].parameters()])
            dot_product = torch.dot(params_i, params_j)
            assert torch.abs(dot_product) < 1e-1, f'Expected dot product to be close to zero, but got {dot_product}'

def test_shift(model, dataset, criterion):
    # Create two random models
    # create dataloader to init loss locus
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    
    point = LossLocus(model, criterion, dataloader)
    direction = LossLocus(model.__class__(), criterion, dataloader)

    # Create a random scalar distance float
    distance = torch.rand(1).item()

    # Call the _shift method
    shifted = Cartographer._shift(point, direction, distance)

    # Assert that the shifted point is not the same object as the original point or direction
    assert id(shifted) != id(point)
    assert id(shifted) != id(direction)

    # Assert that they do not have the same values
    assert not point.equal(shifted)
    assert not direction.equal(shifted)

    # Compute the expected shifted point manually
    expected_shifted = point + direction * distance

    # Assert that the shifted point is equal to the expected shifted point
    assert shifted.equal(expected_shifted)

def test_measure_profiles_serial(cartographer):
    # Execute: Measure the profiles serially
    cartographer.measure_profiles_serial()
    
    # Verify: Check that there are no NaN values in the profiles np.array
    assert not np.isnan(cartographer.profiles).any(), "Profiles should not contain NaN values."

    # all values in profiles[0, :] correspond to the center locus (i.e. the model), so they should be the same
    assert np.allclose(cartographer.profiles[0, :], cartographer.profiles[0, 0]), "All values in the first row should be the same."

def test_measure_profiles_parallel(cartographer):
    # Execute: Measure the profiles in parallel
    cartographer.measure_profiles_parallel()
    
    # Verify: Check that there are no NaN values in the profiles np.array
    assert not np.isnan(cartographer.profiles).any(), "Profiles should not contain NaN values."

    # all values in profiles[0, :] correspond to the center locus (i.e. the model), so they should be the same
    assert np.allclose(cartographer.profiles[0, :], cartographer.profiles[0, 0]), "All values in the first row should be the same."

def test_parallel_vs_serial(cartographer):
    # Execute: Measure the profiles serially
    cartographer.measure_profiles_serial()
    profiles_serial = cartographer.profiles.copy()

    # reset all values in the profiles array to np.non
    cartographer.profiles.fill(np.nan)
    
    # Execute: Measure the profiles in parallel
    cartographer.measure_profiles_parallel()
    profiles_parallel = cartographer.profiles.copy()
    
    # Verify: Check that the profiles are not the same object
    assert id(profiles_serial) != id(profiles_parallel), "Profiles should not be the same object."

    # Verify: Check that the profiles are the same
    assert np.allclose(profiles_serial, profiles_parallel), "Profiles should be the same when measured serially and in parallel."

def test_roughness():
    # Initialize distances_w_0 array
    distances_w_0 = np.array([
        [0, 0],
        [2, 3],
        [4, 6],
        [8, 12],
        [16, 24]
    ])

    # Initialize losses array
    # Adding an extra row for the first entry, which should be the same for all directions
    # Here, we introduce a loss anomaly at the third scale (index 2) for both directions
    anomaly_losses = np.array([
        [1, 1],  # Loss at the center point (index 0)
        [5, 5],  # anomaly in direction 1 and 2
        [1, 1],  
        [1, -5],  # anomaly in direction 2
        [1, 1]
    ])

    anomaly_roughness = Cartographer.roughness(distances_w_0 = distances_w_0, losses = anomaly_losses)

    # the biggest "roughness" should be at in the first direction, at the second scale,
    # because while the loss is the same in both directions, the distance from the center is bigger in the first direction
    # assert that roughness[0,0] > than all other values
    assert np.all(anomaly_roughness[0,0] >= anomaly_roughness[:]), "Roughness is not working as expected"

    # test that linar profiles are have roughness 1
    lin_losses = np.array([
        [0, 0],  # Loss at the center point (index 0)
        [0, 2],  # anomaly in direction 1 and 2
        [0, 4],  
        [0, 8],  # anomaly in direction 2
        [0, 16]
    ])

    lin_roughness = Cartographer.roughness(distances_w_0 = distances_w_0, losses = lin_losses)

    # the roughness of both should be 1. everywhere, 
    assert np.all(lin_roughness == 1), "Roughness should be 1 for linear profiles"

    # test if the function raises an error if the losses array is not the same shape as the distances array
    with pytest.raises(ValueError):
        Cartographer.roughness(distances_w_0 = distances_w_0, losses = lin_losses[:-1])
    # and vice versa
    with pytest.raises(ValueError):
        Cartographer.roughness(distances_w_0 = distances_w_0[:-1], losses = lin_losses)

    # if the first row of losses is not the same, the function should raise an error
    wrong_losses = np.array([
        [2, 1],  # 2 != 1
        [2, 1],
        [2, 1],  
        [2, 1],
        [2, 1]
    ])

    with pytest.raises(ValueError):
        Cartographer.roughness(distances_w_0 = distances_w_0, losses = wrong_losses)

    # if the distances are not always twice as big as the previous row, the function should raise an error
    wrong_distances = np.ones_like(distances_w_0)

    with pytest.raises(ValueError):
        Cartographer.roughness(distances_w_0 = wrong_distances, losses = lin_losses)
    
def test_roughness_in_vivo(cartographer):
    # measure profiles
    cartographer.measure_profiles()
    # calculate roughness
    roughness = cartographer.roughness(distances_w_0=cartographer.distances_w_0, losses=cartographer.profiles)
    # check that the roughness is not nan
    assert not np.isnan(roughness).any(), "Roughness should not contain NaN values."
    

# def test_roughness_measurement(model, dataset, criterion):
#     # Test that the roughness measurement works as expected
#     pass

# def test_full_workflow(model, dataset, criterion):
#     # Test the full Cartographer workflow
#     pass

# def test_plotting(model, dataset, criterion):
#     # Test that the plot generation works as expected
#     pass

# def test_no_grad(model, dataset, criterion):
#     # Test that the gradients are off everywhere.

# def test_dataloader(model, dataset, criterion):

# def test_measure_profiles(model, dataset, criterion):
# see if the decision to use parallel or serial is correct, and the nan check is working