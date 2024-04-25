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
def criterion():
    return torch.nn.CrossEntropyLoss()

def test_initialization(criterion, dataloader):

    A = LossLocus(SimpleCNN(), criterion, dataloader)
    assert isinstance(A, LossLocus), "Model did not initialize correctly."

    with pytest.raises(TypeError):
        LossLocus(model=None, criterion=criterion, dataloader=dataloader)
    with pytest.raises(TypeError):
        LossLocus(model=SimpleCNN(), criterion=criterion, dataloader=None)
    with pytest.raises(TypeError):
        LossLocus(model=SimpleCNN(), criterion=None, dataloader=dataloader)

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


def test_loss(criterion, dataloader):
    model = SimpleCNN()
    A = LossLocus(model, criterion, dataloader)
    
    loss = A.loss()
    assert isinstance(loss, float), "Loss is not a float."
    
    loss2 = A.loss()
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

def test_named_parameters(criterion, dataloader):
    # Test if named_parameters() returns the same parameters as model_w_crit.named_parameters()
    A = LossLocus(SimpleCNN(), criterion, dataloader)
    
    for param, direct_param in zip(A.named_parameters(), A.loss_script.named_parameters()):
        assert param[0] == direct_param[0], "Parameter names do not match."
        assert torch.equal(param[1], direct_param[1]), "Parameter values do not match."

def test_parameters(criterion, dataloader):
    # Test if parameters() returns the same parameters as model_w_crit.parameters()
    A = LossLocus(SimpleCNN(), criterion, dataloader)
        
    for param, direct_param in zip(A.parameters(), A.loss_script.parameters()):
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


def test_normalization(criterion, dataloader):
    A = LossLocus(SimpleCNN(), criterion, dataloader)
    s = torch.rand(1).item()

    A /= abs(A)
    assert pytest.approx(abs(A), abs=1e-6) == 1, "Normalization did not result in a unit norm."

    A *= s
    assert pytest.approx(abs(A), abs=1e-6) == s, "Normalization did not result in the expected norm."

def test_in_place_addition(criterion, dataloader):
    A = LossLocus(SimpleCNN(), criterion, dataloader)
    original_A = A._clone()
    A += original_A
    assert not A.equal(original_A), "In-place addition does not change the model as expected."

def test_in_place_subtraction(criterion, dataloader):
    A = LossLocus(SimpleCNN(), criterion, dataloader)
    zeros_A = A._clone()
    for param in zeros_A.parameters():
        param.data.zero_()
    A -= A  # Subtracting itself to zero out the parameters
    assert A.equal(zeros_A), "In-place subtraction does not zero the model as expected."

def test_in_place_multiplication(criterion, dataloader):
    A = LossLocus(SimpleCNN(), criterion, dataloader)
    zeros_A = A._clone()
    for param in zeros_A.parameters():
        param.data.zero_()
    A *= 0  # Multiplying by 0 to zero out the parameters
    assert A.equal(zeros_A), "In-place multiplication does not zero the model as expected."

def test_division(criterion, dataloader):
    A = LossLocus(SimpleCNN(), criterion, dataloader)
    s = torch.rand(1).item()
    B = A/s
    B *= s
    assert A.equal(B), "Division and multiplication are not inverse operations."

def test_in_place_division(criterion, dataloader):
    A = LossLocus(SimpleCNN(), criterion, dataloader)
    s = torch.rand(1).item()
    B = A*s
    B /= s
    assert A.equal(B), "Division and multiplication are not inverse operations."

def test_error_handling_incompatible_operation(criterion, dataloader):
    A = LossLocus(SimpleCNN(), criterion, dataloader)
    # different architecture: a single linear layer, instead of a CNN
    class MNISTLinearClassifier(nn.Module):
        def __init__(self):
            super(MNISTLinearClassifier, self).__init__()
            self.fc = nn.Linear(784, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten the image to fit Linear layer input
            return self.fc(x)

    B = LossLocus(MNISTLinearClassifier(), criterion, dataloader)
    
    with pytest.raises(ValueError) as exc_info:
        _ = A + B
        
    print(exc_info.value)

