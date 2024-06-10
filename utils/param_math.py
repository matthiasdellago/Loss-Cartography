import torch
from torch import nn
from copy import deepcopy

"""Define utility functions for parameter space arithmetic. iadd is inplace, all others create a new model."""

@torch.no_grad
def iadd(a:nn.Module, b:nn.Module) -> nn.Module:
    """add the parameters of b to a, inplace"""
    for a_param, b_param in zip(a.parameters(), b.parameters()):
        a_param.data.add_(b_param.data)
    return a

@torch.no_grad
def scale(a_old:nn.Module, s:float) -> nn.Module:
    """scale the parameters of a by s"""
    a = deepcopy(a_old)
    for a_param in a.parameters():
        a_param.data.mul_(s)
    return a

@torch.no_grad
def sub(a:nn.Module, b:nn.Module) -> nn.Module:
    """subtract the parameters of b from a"""
    neg_b = scale(b, -1)
    return iadd(neg_b, a)

@torch.no_grad
def norm(a:nn.Module) -> nn.Module:
    """return the norm of the parameters of a"""
    return torch.norm(torch.cat([param.data.flatten() for param in a.parameters()]))

@torch.no_grad
def normalize(a:nn.Module) -> nn.Module:
    """normalize the parameters of a"""
    return scale(a, 1/norm(a))

@torch.no_grad
def rand_like(a: nn.Module) -> nn.Module:
    """normalized random direction in the parameter space of the model."""
    rand = deepcopy(a)
    for param in rand.parameters():
        param.data = torch.randn_like(param.data)
    return normalize(rand)

@torch.no_grad
def project_to_module(a: nn.Module, target_subspace: str) -> nn.Module:
    """
    normalized projection onto the subspace of the model specified by `target_subspace`.
    subspace: Any parameter whose name contains the `target_subspace` string.
    """
    projection = deepcopy(a)
    for name, param in projection.named_parameters():
        if target_subspace not in name:
            param.data.zero_()
    return normalize(projection)