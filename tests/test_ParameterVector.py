# test_ParameterVector.py
import pytest
import torch
from torch.nn import Parameter
from ParameterVector import ParameterVector

class TestModel(ParameterVector):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.ln = torch.nn.LayerNorm([4, 4, 4])
        self.fc1 = torch.nn.Linear(4 * 4 * 4, 10)
        self.bias = Parameter(torch.randn(10))
        self.dropout = torch.nn.Dropout(0.1)
        self.fc2 = torch.nn.Linear(10, 2)

def test_initialization():
    A = TestModel()
    assert isinstance(A, TestModel), "Model did not initialize correctly."

def test_equality():
    A = TestModel()
    B = TestModel()

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
    with torch.no_grad():
        C.fc1.weight.add_(torch.randn_like(C.fc1.weight))
    
    assert not A.equal(C), "Model is considered equal to its modified clone."
    assert not C.equal(A), "Modified clone is considered equal to the original model."

def test_vector_space_axioms():
    A = TestModel()
    B = TestModel()
    C = TestModel()
    s = torch.rand(1).item()

    # Testing commutativity of addition
    assert (A + B).equal(B + A), "Addition is not commutative."

    # Testing associativity of addition
    assert ((A + B) + C).equal(A + (B + C)), "Addition is not associative."

    # Testing distributivity of scalar multiplication over vector addition
    assert ((A + B) * s).equal((A * s) + (B * s)), "Distributivity of scalar multiplication over addition does not hold."

def test_triangle_inequality():
    A = TestModel()
    B = TestModel()
    assert abs(A + B) <= abs(A) + abs(B), "Triangle inequality does not hold."


def test_normalization():
    A = TestModel()
    s = torch.rand(1).item()

    A *= (1 / abs(A))
    assert pytest.approx(abs(A), abs=1e-6) == 1, "Normalization did not result in a unit norm."

    A *= s
    assert pytest.approx(abs(A), abs=1e-6) == s, "Normalization did not result in the expected norm."

def test_in_place_addition():
    A = TestModel()
    original_A = A._clone()
    A += original_A
    assert not A.equal(original_A), "In-place addition does not change the model as expected."

def test_in_place_subtraction():
    A = TestModel()
    zeros_A = A._clone()
    for param in zeros_A.parameters():
        param.data.zero_()
    A -= A  # Subtracting itself to zero out the parameters
    assert A.equal(zeros_A), "In-place subtraction does not zero the model as expected."

def test_in_place_multiplication():
    A = TestModel()
    zeros_A = A._clone()
    for param in zeros_A.parameters():
        param.data.zero_()
    A *= 0  # Multiplying by 0 to zero out the parameters
    assert A.equal(zeros_A), "In-place multiplication does not zero the model as expected."

def test_error_handling_incompatible_operation():
    A = TestModel()
    B = TestModel()
    # Intentionally making models incompatible
    B.fc2 = torch.nn.Linear(10, 3)  # Changing output features to make it incompatible
    
    with pytest.raises(ValueError):
        _ = A + B