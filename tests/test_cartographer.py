#test_cartographer.py
import pytest
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from cartographer import Cartographer
from simple_cnn import SimpleCNN
import numpy as np

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def model(device):
    return SimpleCNN().to(device)

@pytest.fixture
def dataloader():
    # Use a standard MNIST normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    return DataLoader(dataset, batch_size=64, shuffle=False)

@pytest.fixture
def loss_function():
    return torch.nn.CrossEntropyLoss()

def test_validate_inputs(model, dataloader, loss_function):
    # Test validation during __init__ 
    # Test that the Cartographer class validates the input arguments correctly
    with pytest.raises(ValueError):
        Cartographer(model=None, dataloader=dataloader, loss_function=loss_function)
    with pytest.raises(ValueError):
        Cartographer(model=model, dataloader=None, loss_function=loss_function)
    with pytest.raises(ValueError):
        Cartographer(model=model, dataloader=dataloader, loss_function=None)
    with pytest.raises(ValueError):
        Cartographer(model=model, dataloader=dataloader, loss_function=loss_function, num_directions=-1)
    with pytest.raises(ValueError):
        Cartographer(model=model, dataloader=dataloader, loss_function=loss_function, pow_min_dist=3, pow_max_dist=1)
    
    # TODO: These validation tests are not excellent. Improve or remove them.

    # Test operational readiness of model
    with pytest.raises(ValueError, match="Model does not support evaluation mode"):
        class BrokenModel(nn.Module):
            def eval(self):
                raise AttributeError("Cannot set evaluation mode")
        broken_model = BrokenModel()
        Cartographer(model=broken_model, dataloader=dataloader, loss_function=loss_function)

    # Test model and dataloader compatibility
    with pytest.raises(ValueError, match="Model failed to process input from DataLoader"):
        class IncompatibleModel(nn.Module):
            def forward(self, x):
                raise ValueError("Invalid input")
        incompatible_model = IncompatibleModel()
        Cartographer(model=incompatible_model, dataloader=dataloader, loss_function=loss_function)

    # Test target and output compatibility with loss function
    with pytest.raises(ValueError, match="Loss function cannot process model output"):
        class OutputMismatchModel(nn.Module):
            def forward(self, x):
                return x + 1.  # Assuming non-compatible output
        mismatch_model = OutputMismatchModel()
        Cartographer(model=mismatch_model, dataloader=dataloader, loss_function=loss_function)


    # Test that the Cartographer class accepts the input arguments correctly
    cartographer = Cartographer(model=model, dataloader=dataloader, loss_function=loss_function)
    assert cartographer.center is model
    assert cartographer.dataloader is dataloader
    assert cartographer.loss_function is loss_function

def test_distance_generation_1(model, dataloader, loss_function):
    # Test that the distance generation works as expected
    # If num_directions is 1 and pow_min_dist and pow_max_dist are 0, the distances should be of shape (1, 1) and contain only 1.0
    cartographer = Cartographer(model=model, dataloader=dataloader, loss_function=loss_function, num_directions=1, pow_min_dist=0, pow_max_dist=0)
    distances = cartographer.generate_distances()

    assert isinstance(distances, np.ndarray), f"Expected type numpy.ndarray, but got {type(distances)}"
    assert distances.shape == (1, 1), f"Unexpected shape: {distances.shape}, expected (1, 1)"
    assert np.isclose(distances[0,0], 1.0), f"Unexpected value at (0,0): {distances[0,0]}, expected 1.0"

def test_distance_generation_manual(model, dataloader, loss_function):
    # Test that the distance generation works as expected
    # Caclculate distances by hand for num_directions=2, pow_min_dist=0 and pow_max_dist = 2
    cartographer = Cartographer(model=model, dataloader=dataloader, loss_function=loss_function, num_directions=2, pow_min_dist=0, pow_max_dist=2)
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

def test_direction_generation(model, dataloader, loss_function):
    num = 7
    cartographer = Cartographer(model=model, dataloader=dataloader, loss_function=loss_function, num_directions=num)
    directions = cartographer.generate_directions()
    
    assert isinstance(directions, nn.ModuleList), f"Expected type nn.ModuleList, but got {type(directions)}"
    # check that all objects in the ModuleList are of the same type as model
    assert all(isinstance(direction, type(model)) for direction in directions), f"Expected all directions to be of type {type(model)}, but got {set(type(direction) for direction in directions)}"
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

def test_shift(model):
    # Create two random ParameterVector objects
    point = model
    direction = model.__class__()

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

def test_measure_loss(model, dataloader, loss_function):
    """
    Test to ensure that the loss measured by the _measure_loss method is deterministic
    under the same conditions.
    TODO: more tests? What else can we test here? Maybe train a model and make sure that the loss is lower than the initial loss..
    """
    # Measure the loss for the first run
    loss_first_run = Cartographer._measure_loss(model, dataloader, loss_function)
    
    # Measure the loss for the second run
    loss_second_run = Cartographer._measure_loss(model, dataloader, loss_function)
    
    # Check that the losses from the two runs are the same
    assert loss_first_run == loss_second_run, f"Loss should be deterministic, but yielded {loss_first_run}, and {loss_second_run} for the same inputs."    

# def test_location_generation(model, dataloader, loss_function):
#     # Test that the location generation works as expected
#     pass

# def test_loss_measurement(model, dataloader, loss_function):
#     # Test that the loss measurement works as expected
#     pass

# def test_roughness_measurement(model, dataloader, loss_function):
#     # Test that the roughness measurement works as expected
#     pass

# def test_full_workflow(model, dataloader, loss_function):
#     # Test the full Cartographer workflow
#     pass

# def test_plot_generation(model, dataloader, loss_function):
#     # Test that the plot generation works as expected
#     pass