# test_loss_locus.py
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.jit as jit
from torchvision import transforms
from torchvision.datasets import MNIST
from simple_cnn import SimpleCNN
from loss_locus import LossLocus
import datetime # used to break jit.scriptability for testing
import numpy as np

@pytest.fixture(scope="session")
def dataloader():
    # Use a standard MNIST normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    # Specify the size of the subset you want, e.g., 100 samples
    subset_size = 100

    # Generate a random subset of indices
    indices = np.random.choice(len(dataset), subset_size, replace=False)

    # Create the subset dataset
    subset_dataset = torch.utils.data.Subset(dataset, indices)

    return DataLoader(subset_dataset, batch_size=64, shuffle=False)

@pytest.fixture(scope="session")
def criterion():
    return torch.nn.CrossEntropyLoss()

@pytest.fixture(scope="session")
def model():
    return SimpleCNN()

@pytest.fixture(scope="session")
def losslocus(model, criterion, dataloader):
    return LossLocus(model, criterion, dataloader)

def test_data_loading_and_type(dataloader):
    data, targets = next(iter(dataloader))

    assert data.dtype == torch.float32, f"Data type should be float32, got {data.dtype}"
    assert targets.dtype == torch.int64, f"Target type should be int64, got {targets.dtype}"

def test_initialization(model,criterion, dataloader):

    A = LossLocus(model, criterion, dataloader)
    assert isinstance(A, LossLocus), "Model did not initialize correctly."

    with pytest.raises(TypeError):
        LossLocus(model=None, criterion=criterion, dataloader=dataloader)
    with pytest.raises(TypeError):
        LossLocus(model=model, criterion=criterion, dataloader=None)
    with pytest.raises(TypeError):
        LossLocus(model=model, criterion=None, dataloader=dataloader)

    # test non-jit.scriptable model
    class NonScriptableModel(nn.Module):
        def __init__(self):
            super(NonScriptableModel, self).__init__()
            self.fc = nn.Linear(784, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            # add a non-scriptable operation, datetime
            x = x + datetime.datetime.now().microsecond
            return self.fc(x)

    # check that it is indeed not scriptable and raises an exception
    with pytest.raises(Exception):
        jit.script(NonScriptableModel())

    # check that the warning is raised when the model is not scriptable
    with pytest.warns(UserWarning):
        scriptless = LossLocus(NonScriptableModel(), criterion, dataloader)
        assert isinstance(scriptless, LossLocus), "Model is {type(scriptless)}, expected LossLocus."
        assert isinstance(scriptless.loss_script, nn.Module), "Model is {type(scriptless.loss_script)}, expected nn.Module, because the model is not scriptable."

    # test broken dataloader and model combination
    with pytest.raises(ValueError):
        _ = LossLocus(nn.Linear(10,10), criterion, dataloader)

def test_loss_script(losslocus, dataloader):
    assert isinstance(losslocus.loss_script, jit.ScriptModule), "loss_script is not an instance of jit.ScriptModule."

    # Test if the loss_script is callable
    for data, target in dataloader:
        _ = losslocus.loss_script(data, target)
        break
    
def test_loss(model, criterion, dataloader):
    losslocus = LossLocus(model, criterion, dataloader) # create a new instance because fixtures can't be called

    loss = losslocus.loss()
    assert isinstance(loss, np.longdouble), "Loss is not longdouble."
    
    loss2 = losslocus.loss()
    assert pytest.approx(loss, abs=1e-6) == loss2, "Loss calculation is not deterministic."

    # calculate the loss manually
    total_loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            total_loss += criterion(model(data), target).item()
    manual_loss = total_loss / len(dataloader)
    assert pytest.approx(loss, abs=1e-6) == manual_loss, "Loss calculation is incorrect. Expected: {manual_loss}, got: {loss}"

# def test_script_speedup(criterion, dataloader):
#     assert False, "Test not implemented." # TODO: Implement test

def test_named_parameters(losslocus):
    # Test if named_parameters() returns the same parameters as model_w_crit.named_parameters()
    for param, direct_param in zip(losslocus.named_parameters(), losslocus.loss_script.named_parameters()):
        assert param[0] == direct_param[0], "Parameter names do not match."
        assert torch.equal(param[1], direct_param[1]), "Parameter values do not match."

def test_parameters(losslocus):
    # Test if parameters() returns the same parameters as model_w_crit.parameters()
    for param, direct_param in zip(losslocus.parameters(), losslocus.loss_script.parameters()):
        assert torch.equal(param, direct_param), "Parameter values do not match."

def test_equality(criterion, dataloader):
    A = LossLocus(SimpleCNN(), criterion, dataloader)
    B = LossLocus(SimpleCNN(), criterion, dataloader)

    # Newly initialized models with random parameters should not be equal
    assert not A.equal(B), "Models with different initial parameters are considered equal."
    assert not B.equal(A), "Models with different initial parameters are considered equal."

    # A model should always be equal to itself
    assert A.equal(A), "Model is not equal to itself."

    # Cloning and verifying equality
    C = A._clone()
    assert A.equal(C), "Cloned model is not considered equal to the original."
    assert C.equal(A), "Original model is not considered equal to the clone."
    
    # Modifying the clone and verifying inequality
    # Get the first parameter of the model
    first_param = next(A.parameters())
    # Modify the first parameter
    first_param.data += 1.
    
    assert not A.equal(C), "Model is considered equal to its modified clone."
    assert not C.equal(A), "Modified clone is considered equal to the original model."

def test_vector_space_axioms(criterion, dataloader):
    A = LossLocus(SimpleCNN(), criterion, dataloader)
    B = LossLocus(SimpleCNN(), criterion, dataloader)
    C = LossLocus(SimpleCNN(), criterion, dataloader)
    s = torch.rand(1).item()

    # Testing commutativity of addition
    assert (A + B).equal(B + A), "Addition is not commutative."

    # Testing associativity of addition
    assert ((A + B) + C).equal(A + (B + C)), "Addition is not associative."

    # Testing distributivity of scalar multiplication over vector addition
    assert ((A + B) * s).equal((A * s) + (B * s)), "Distributivity of scalar multiplication over addition does not hold."

def test_triangle_inequality(criterion, dataloader):
    A = LossLocus(SimpleCNN(), criterion, dataloader)
    B = LossLocus(SimpleCNN(), criterion, dataloader)
    assert abs(A + B) <= abs(A) + abs(B), "Triangle inequality does not hold."


def test_normalization(losslocus):
    s = torch.rand(1).item()

    losslocus /= abs(losslocus)
    assert pytest.approx(abs(losslocus), abs=1e-6) == 1, "Normalization did not result in a unit norm."

    losslocus *= s
    assert pytest.approx(abs(losslocus), abs=1e-6) == s, "Normalization did not result in the expected norm."

def test_in_place_addition(losslocus):
    original_A = losslocus._clone()
    losslocus += original_A
    assert not losslocus.equal(original_A), "In-place addition does not change the model as expected."

def test_in_place_subtraction(losslocus):
    zeros_locus = losslocus._clone()
    for param in zeros_locus.parameters():
        param.data.zero_()
    losslocus -= losslocus  # Subtracting itself to zero out the parameters
    assert losslocus.equal(zeros_locus), "In-place subtraction does not zero the model as expected."

def test_in_place_multiplication(losslocus):
    zeros_locus = losslocus._clone()
    for param in zeros_locus.parameters():
        param.data.zero_()
    losslocus *= 0  # Multiplying by 0 to zero out the parameters
    assert losslocus.equal(zeros_locus), "In-place multiplication does not zero the model as expected."

def test_division(losslocus):
    s = torch.rand(1).item()
    B = losslocus/s
    B *= s
    assert losslocus.equal(B), "Division and multiplication are not inverse operations."

def test_in_place_division(losslocus):
    s = torch.rand(1).item()
    B = losslocus*s
    B /= s
    assert losslocus.equal(B), "Division and multiplication are not inverse operations."

def test_error_handling_incompatible_operation(criterion, dataloader, losslocus):
    CNNlocus = losslocus
    # different architecture: a single linear layer, instead of a CNN
    class MNISTLinearClassifier(nn.Module):
        def __init__(self):
            super(MNISTLinearClassifier, self).__init__()
            self.fc = nn.Linear(784, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten the image to fit Linear layer input
            return self.fc(x)

    Linearlocus = LossLocus(MNISTLinearClassifier(), criterion, dataloader)
    
    with pytest.raises(ValueError):
        _ = CNNlocus + Linearlocus

def test_random_direction(losslocus):
    A = losslocus.random_direction()
    B = losslocus.random_direction()
    
    assert not A.equal(losslocus), "Random parameters are equal to the original model."
    assert not A.equal(B), "Random parameters are not unique."
    assert pytest.approx(abs(A), abs=1e-6) == 1, "Random parameters are not normalised."
    # check that the new parameters are randomly distributed between -1 and 1
    for param in A.parameters():
        assert torch.all(param >= -1) and torch.all(param <= 1), f"Random parameters {param} are not between -1 and 1."
        assert torch.any(param > 0), f"There are no positive parameters in {param}."
        assert torch.any(param < 0), f"There are no negative parameters in {param}."
