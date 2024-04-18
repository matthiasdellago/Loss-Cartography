#parameter_vector.py
import torch
import torch.nn as nn
from copy import deepcopy

class ParameterVector(nn.Module):
    """
    Extends nn.Module to provide a simple interface for manipulating a model parameter with vector operations.
    The parameter space becomes a vector space.
    Watch out:
        - All operations can be done in-place for resource efficiency, but non-in-place versions are also available.
        - There is no gradient tracking.
        - There are in-place and non-in-place versions of the operations. The in-place versions are preferred for resource efficiency.
        - Use equal() instead of == or != if you want to see if the parameters are the same.
    TODO: Find a nice way to make an existing nn.Module compatible with this class.
    """

    def _ensure_compatible(self, other: 'ParameterVector') -> None:
        if not isinstance(other, self.__class__):
            raise TypeError(f"Compatibility error: 'other' model {other} is of type {type(other)}, expected {type(self)}.")
        
        for (name, param), (other_name, other_param) in zip(self.named_parameters(), other.named_parameters()):
            if name != other_name:
                raise ValueError(f"Parameter name mismatch: {name} (self) vs {other_name} (other).")
            if param.data.shape != other_param.data.shape:
                raise ValueError(f"Parameter shape mismatch in '{name}': {param.data.shape} (self) vs {other_param.data.shape} (other).")

    def _clone(self) -> 'ParameterVector':
        """
        Helper method to clone this ParameterVector, creating a deep copy.
        """
        return deepcopy(self)

    def __iadd__(self, other: 'ParameterVector') -> 'ParameterVector':
        """
        Performs in-place element-wise addition between this ParameterVector and another.
        This is the preferred method for resource efficiency.
        """
        self._ensure_compatible(other)
        for (name, param), (_, other_param) in zip(self.named_parameters(), other.named_parameters()):
            param.data.add_(other_param.data)
        return self

    def __isub__(self, other: 'ParameterVector') -> 'ParameterVector':
        """
        Performs in-place element-wise subtraction between this ParameterVector and another.
        This is the preferred method for resource efficiency.
        """
        self._ensure_compatible(other)
        for (name, param), (_, other_param) in zip(self.named_parameters(), other.named_parameters()):
            param.data.sub_(other_param.data)
        return self

    def __imul__(self, other: 'ParameterVector') -> 'ParameterVector':
        """
        Performs in-place element-wise multiplication between this ParameterVector's parameters and a scalar.
        This is the preferred method for resource efficiency.
        """
        if not isinstance(other, (int, float)):
            raise TypeError(f"Multiplication error: 'other' is of type {type(other)}, expected int or float.")
        for name, param in self.named_parameters():
            param.data.mul_(other)
        return self

    def __abs__(self) -> float:
        """
        Returns the L2 norm of the ParameterVector.
        """
        norm = 0
        for param in self.parameters():
            norm += torch.norm(param).item() ** 2
        return norm ** 0.5

    def equal(self, other, tol=1e-6) -> bool:
        """
        Compares this ParameterVector with another for equality, within a tolerance.
        Not overloading __eq__ because that opens a can of worms with inheritance of __hash__.
        """
        self._ensure_compatible(other)
        for param_self, param_other in zip(self.parameters(), other.parameters()):
            if not torch.allclose(param_self, param_other, atol=tol):
                return False
        return True

    def __add__(self, other: 'ParameterVector') -> 'ParameterVector':
        """
        Performs element-wise addition between this ParameterVector and another, returning a new ParameterVector
        without modifying the original ones. This operation is not in-place.
        """
        result = self._clone()
        result += other  # Utilizes the __iadd__ for in-place addition on the clone
        return result

    def __sub__(self, other: 'ParameterVector') -> 'ParameterVector':
        """
        Performs element-wise subtraction between this ParameterVector and another, returning a new ParameterVector
        without modifying the original ones. This operation is not in-place.
        """
        result = self._clone()
        result -= other  # Utilizes the __isub__ for in-place subtraction on the clone
        return result

    def __mul__(self, other: 'ParameterVector') -> 'ParameterVector':
        """
        Performs element-wise multiplication between this ParameterVector's parameters and a scalar, returning a new ParameterVector
        without modifying the original. This operation is not in-place.
        """
        result = self._clone()
        result *= other  # Utilizes the __imul__ for in-place multiplication on the clone
        return result


