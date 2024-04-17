#test_carographer.py
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from cartographer import Cartographer
from simple_cnn import SimpleCNN

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

def test_input_validation(model, dataloader, loss_function):
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
        Cartographer(model=model, dataloader=dataloader, loss_function=loss_function, num_scales=-1)
    
    # Test that the Cartographer class accepts the input arguments correctly
    cartographer = Cartographer(model=model, dataloader=dataloader, loss_function=loss_function)
    assert cartographer.center is model
    assert cartographer.dataloader is dataloader
    assert cartographer.loss_function is loss_function

# def test_direction_generation(model, dataloader, loss_function):
#     # Test that the direction generation works as expected
#     pass

# def test_distance_generation(model, dataloader, loss_function):
#     # Test that the distance generation works as expected
#     pass

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