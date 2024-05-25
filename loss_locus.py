#loss_locus.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch import jit
from warnings import warn
from copy import deepcopy
import numpy as np

class ModelwithCriterion(nn.Module):
    """
    Helper class: Wraps a model and a loss function into a single module.
    """
    def __init__(self, model: nn.Module, criterion: _Loss) -> None:
        super(ModelwithCriterion, self).__init__()
        self.model = model
        # double precision. otherwise we get precision errors in the loss landscape round e-7.
        self.criterion = criterion.double()
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # convert forward pass to double precision
        output = self.model(x).double()
        return self.criterion(output, target)

class LossLocus():
    """
    Wraps a model, a loss function, and a dataloader.
    Provides a simple interface for manipulating a model parameter with vector operations and measuring loss.
    The parameter space becomes a vector space. The loss function becomes a function on this vector space.
    Watch out:
        - All operations can be done in-place for resource efficiency, but non-in-place versions are also available.
        - There is no guarantee that this works with grandients.
        - There are in-place and non-in-place versions of the operations. The in-place versions are preferred for resource efficiency.
        - Use equal() instead of == or != if you want to see if the parameters are the same.
    TODO: ensure eval, nograd, and device are set correctly. Where is the best place to do this???
    """

    def __init__(self, model: nn.Module, criterion: _Loss, dataloader: DataLoader) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # validate inputs
        if not isinstance(model, nn.Module):
            raise TypeError(f"Model error: 'model' is of type {type(model)}, expected nn.Module.")
        if not isinstance(criterion, _Loss):
            raise TypeError(f"Loss error: 'criterion' is of type {type(criterion)}, expected torch.nn.modules.loss._Loss.")
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f"Dataloader error: 'dataloader' is of type {type(dataloader)}, expected torch.utils.data.DataLoader.")
        self.dataloader = dataloader
        
        # see if the model, and the loss function are jit.scriptable
        try:
            jit.script(model)
        except Exception as e:
            warn("Model jit.scriptable. Performance may be affected. Error: {e}")
        try:
            jit.script(criterion)
        except Exception as e:
            warn("Criterion not jit.scriptable. Performance may be affected. Error: {e}")
        
        model_w_crit = ModelwithCriterion(model, criterion)

        try:
            self.loss_script = jit.script(model_w_crit)
        except Exception as e:
            warn("Model and criterion combined are not jit.scriptable. Performance may be affected. Error: {e}")
            # if the model is not jit.scriptable, fall back to the non-jit.scriptable version
            self.loss_script = model_w_crit

        # move the model to the device
        self.loss_script.to(device)

        # test if the model can process the input from the DataLoader with a sample
        try:
            # get the first data and target, but don't remove it from the DataLoader
            for data, target in self.dataloader:
                data, target = data.to(device), target.to(device)
                self.loss_script(data, target)
                break # only need to test one
        except Exception as e:
            raise ValueError("Model failed to process input from DataLoader. Error: {e}")

        #  set the model to eval mode
        self.loss_script.eval()

    def to(self, device: torch.device) -> None:
        """
        Moves the model to the given device.
        """
        self.loss_script.to(device)

    def loss(self) -> float:
        """
        Measures the loss of the model on the entire dataset.
        longdouble to avoid all precision errors.
        """
        average_loss = np.longdouble(0.)
        for data, target in self.dataloader:
            # load the data and target to the device
            device = next(self.loss_script.parameters()).device
            data = data.to(device)
            target = target.to(device)
            
            loss = np.longdouble(self.loss_script(data,target).item())
            # divide before adding the losses together, for numerical stability
            average_loss += loss/len(self.dataloader)
        
        return average_loss

    
    def named_parameters(self) -> [(str, torch.Tensor)]:
        """
        Map LossLocus.named_parameters() to model.named_parameters().
        """
        return self.loss_script.named_parameters()

    def parameters(self) -> [torch.Tensor]:
        """
        Map LossLocus.parameters() to model.parameters().
        """
        return self.loss_script.parameters()


    def _ensure_compatible(self, other: 'LossLocus') -> None:
        if not isinstance(other, self.__class__):
            raise TypeError(f"Compatibility error: 'other' model {other} is of type {type(other)}, expected {type(self)}.")
        
        for (name, param), (other_name, other_param) in zip(self.named_parameters(), other.named_parameters()):
            if name != other_name:
                raise ValueError(f"Parameter name mismatch: {name} (self) vs {other_name} (other).")
            if param.data.shape != other_param.data.shape:
                raise ValueError(f"Parameter shape mismatch in '{name}': {param.data.shape} (self) vs {other_param.data.shape} (other).")

    def _clone(self) -> 'LossLocus':
        """
        Helper method to clone this LossLocus, creating a deep copy.
        """
        return deepcopy(self)

    def __iadd__(self, other: 'LossLocus') -> 'LossLocus':
        """
        Performs in-place element-wise addition between this LossLocus and another.
        This is the preferred method for resource efficiency.
        """
        self._ensure_compatible(other)
        for param, other_param in zip(self.parameters(), other.parameters()):
            param.data.add_(other_param.data)
        return self

    def __isub__(self, other: 'LossLocus') -> 'LossLocus':
        """
        Performs in-place element-wise subtraction between this LossLocus and another.
        This is the preferred method for resource efficiency.
        """
        self._ensure_compatible(other)
        for param, other_param in zip(self.parameters(), other.parameters()):
            param.data.sub_(other_param.data)
        return self

    def __imul__(self, scalar: (int, float)) -> 'LossLocus':
        """
        Performs in-place element-wise multiplication between this LossLocus's parameters and a scalar.
        This is the preferred method for resource efficiency.
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Multiplication error: 'scalar' is of type {type(scalar)}, expected int or float.")
        for name, param in self.named_parameters():
            param.data.mul_(scalar)
        return self

    def __itruediv__(self, scalar: (int, float)) -> 'LossLocus':
        """
        Performs in-place element-wise division between this LossLocus's parameters and a scalar.
        This is the preferred method for resource efficiency.
        Defers to __imul__ for in-place multiplication by the reciprocal.
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Division error: 'scalar' is of type {type(scalar)}, expected int or float.")

        return self.__imul__(1/scalar)

    def __abs__(self) -> float:
        """
        Returns the L2 norm of the LossLocus.
        """
        norm = 0
        for param in self.parameters():
            norm += torch.norm(param).item() ** 2
        return norm ** 0.5

    def equal(self, other: 'LossLocus', tol=1e-8) -> bool:
        """
        Compares this LossLocus with another for equality, within a tolerance.
        Not overloading __eq__ because that opens a can of worms with inheritance of __hash__.
        """
        self._ensure_compatible(other)
        for param, other_param in zip(self.parameters(), other.parameters()):
            if not torch.allclose(param, other_param, atol=tol):
                return False
        return True

    def __add__(self, other: 'LossLocus') -> 'LossLocus':
        """
        Performs element-wise addition between this LossLocus and another, returning a new LossLocus
        without modifying the original ones. This operation is not in-place.
        """
        result = self._clone()
        result += other  # Utilizes the __iadd__ for in-place addition on the clone
        return result

    def __sub__(self, other: 'LossLocus') -> 'LossLocus':
        """
        Performs element-wise subtraction between this LossLocus and another, returning a new LossLocus
        without modifying the original ones. This operation is not in-place.
        """
        result = self._clone()
        result -= other  # Utilizes the __isub__ for in-place subtraction on the clone
        return result

    def __mul__(self, scalar: (int, float)) -> 'LossLocus':
        """
        Performs element-wise multiplication between this LossLocus's parameters and a scalar, returning a new LossLocus
        without modifying the original. This operation is not in-place.
        """
        result = self._clone()
        result *= scalar  # Utilizes the __imul__ for in-place multiplication on the clone
        return result
    
    def __truediv__(self, scalar: (int, float)) -> 'LossLocus':
        """
        Performs element-wise division between this LossLocus's parameters and a scalar, returning a new LossLocus
        without modifying the original. This operation is not in-place.
        """
        result = self._clone()
        result /= scalar  # Utilizes the __idiv__ for in-place division on the clone
        return result

    def random_direction(self) -> 'LossLocus':
        """
        Returns a random, normalised instance of LossLocus in the same parameter space.
        """
        result = self._clone()
        # randomise the parameters
        for param in result.parameters():
            param.data = torch.randn_like(param.data)
        # normalise the parameters
        result /= abs(result)
        return result
    
    def size(self) -> int:
        """
        Returns the number of bytes used by the parameters of the model.
        """
        size = 0
        for param in self.parameters():
            size += param.data.element_size() * param.data.numel()
        return size