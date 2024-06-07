# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import os

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

center = SimpleMLP()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MNIST(root='./data', download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))
BATCH_SIZE = 100
num_workers = os.cpu_count()  # Adjust based on available CPU cores
pin_memory = torch.cuda.is_available()  # Enable pinning if using CUDA
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)

# criterion functional
criterion = F.cross_entropy

# Do one SGD step to get the gradient?
GRAD = False
if not GRAD:
    torch.set_grad_enabled(False)

# parameter subspaces to investigate.
# for each string, the union of all modules that contain the string in their name will be considered.
# we will sample a random direction in that subspace + the gradient ascent and descent directions projected onto that subspace.
SUBSPACES = ['fc1', 'weight', 'bias']

#check if we are on GPU, otherwise just proof of concept
if torch.cuda.is_available():
    NUM_DIRS = 20
    MAX_OOM = 2
    MIN_OOM = -13 # go down until all roughness disappears due to numerical precision.
else:
    NUM_DIRS = 0 # number of random directions to sample, in addition to the gradient ascent + descent, and radially in + out.
    MAX_OOM = 0 # maximum order of magnitude to sample
    MIN_OOM = -1 # minimum order of magnitude to sample

print(f'Running on {device}')

# %% 
# Define utility functions for parameter space arithmetic. iadd is inplace, all others create a new model.

from copy import deepcopy

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
def abs(a:nn.Module) -> nn.Module:
    """return the norm of the parameters of a"""
    return torch.norm(torch.cat([param.data.flatten() for param in a.parameters()]))

@torch.no_grad
def normalize(a:nn.Module) -> nn.Module:
    """normalize the parameters of a"""
    return scale(a, 1/abs(a))

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

# %%
# Define profiling context manager: measure execution time and GPU RAM usage before and after.

from time import perf_counter
from contextlib import contextmanager
import torch
import psutil

@contextmanager
def profiler(description: str, length: int = 80, pad_char: str = ':') -> None:
    def print_memory_usage(prefix):
        if torch.cuda.is_available():
            print(f'{prefix} GPU RAM: Allocated: {torch.cuda.memory_allocated() / 1e9:.2g} GB | Reserved: {torch.cuda.memory_reserved() / 1e9:.2g} GB')
        print(f'{prefix} RAM: Used: {psutil.virtual_memory().used / 1e9:.2g} GB | Available: {psutil.virtual_memory().available / 1e9:.2g} GB')

    print('\n' + description.center(length, pad_char))
    print_memory_usage('Before:')
    start = perf_counter()
    yield

    seconds = perf_counter() - start
    print(f'>>> {seconds:.2f} s to execute')
    print_memory_usage('After: ')
    print(f'Finished {description} in {seconds:.2f} s'.center(length, pad_char))


# %%

radial = normalize(center)

dirs = {
    'Radially Out': radial,
    'Radially In': scale(radial,-1),
    }

# add random directions
dirs.update({f'Random {i}': rand_like(center) for i in range(NUM_DIRS)})

def ascent_direction(model:nn.Module, criterion:nn.functional, dataloader:DataLoader) -> nn.Module:
    """Find the direction of gradient ascent for a given model, and return it as a normalized model."""
    # create a model to find the gradient
    grad_model = deepcopy(model).to(device)
    optimizer = torch.optim.SGD(grad_model.parameters(), lr=1)
    
    torch.set_grad_enabled(True)
    # gradient accumulate over the whole dataset
    with profiler('1 step of gradient descent over the whole dataset'):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = grad_model(data)
            loss = criterion(output, target)
            loss.backward()

    optimizer.step()
    # IMPORTANT: from this point on we won't need any gradients
    torch.set_grad_enabled(False)

    # calculate the direction of gradient decent from the model
    grad_model = grad_model.to('cpu')
    dir_ascent = sub(model, grad_model) # from updated model to original model, back up the gradient
    return normalize(dir_ascent)

if GRAD:
    dir_ascent = ascent_direction(center, criterion, dataloader)
    dir_descent = scale(dir_ascent, -1) # descent = -1*ascent
    dirs.update({'Ascent': dir_ascent,'Descent': dir_descent})

# add projections of each dir in dirs onto each SUBSPACE
for dir_name, direction in list(dirs.items()):
    for subspace in SUBSPACES:
        projected_dir = project_to_module(direction, subspace)
        new_key = f'{dir_name} ⋅ P({subspace})'
        dirs[new_key] = projected_dir

# check that they are all normalized
assert all(torch.isclose(abs(d), torch.tensor(1.0)) for d in dirs.values())

# %%
import pandas as pd
import numpy as np

def dirs_and_dists(dir_names:[str], MIN_OOM:int, MAX_OOM:int) -> pd.DataFrame:
    """
    Create a DataFrame with a MultiIndex of direction names and doubling steps,
    containing logarithmically spaced 'Distance' values from 10^MIN_OOM to just over 10^MAX_OOM.
    """
    # How often do I need to double 10^MIN_OOM to reach 10^MAX_OOM?
    doublings = int(np.ceil((MAX_OOM - MIN_OOM) * np.log2(10)))
    steps = np.arange(doublings)

    # Compute all the distances we need to sample.
    dists = 2**steps * 10**MIN_OOM
    
    # In order to sample the spaces between 2^scale and 2^(scale+1),
    # and uniformly sample the log-space, we multiply each direction by a different offset
    offset_per_dir = np.logspace(base=2, start=0, stop=1, num=len(dir_names), endpoint=False)

    # Compute the outer product of all offsets and distances.
    dists_per_dir = np.outer(offset_per_dir, dists)

    # Create an 'outer product' of the corresponding indeces using pd.MultiIndex.from_product
    multi_index = pd.MultiIndex.from_product([dir_names, steps+1], names=['Direction', 'Step'])
    
    # Create DataFrame directly with the computed distances.
    return pd.DataFrame({'Distance': dists_per_dir.flatten()}, index=multi_index)

df = dirs_and_dists(dirs.keys(), MIN_OOM, MAX_OOM)

# %%

# fill the dataframe with the models
for (dir_name, step), distance in df['Distance'].items():
    direction = dirs[dir_name]
    shift = scale(direction,distance) # shift = direction * distance
    location = iadd(shift, center)    # location = center + shift
    df.at[(dir_name, step), 'Model'] = location 


# %%
# Parallel evaluation with vmap()

from torch.func import stack_module_state, functional_call
from torch import vmap

# add the center model
ensemble_list = [center] + list(df['Model'])
df.drop(columns='Model', inplace=True)

def eval_ensemble(ensemble_list:[nn.Module], dataloader:DataLoader, criterion:nn.functional, device:str) -> torch.Tensor:

    with profiler(f'Ensemble size: {len(ensemble_list)}', pad_char='-'):

        ensemble_size = len(ensemble_list)

        [model.to(device) for model in ensemble_list]
        
        # stack to prepare for vmap
        stacked_ensemble = stack_module_state(ensemble_list)
        del ensemble_list

        # Construct a "stateless" version of one of the models. It is "stateless" in
        # the sense that the parameters are meta Tensors and do not have storage.
        meta_model = deepcopy(center).to('meta')

        def meta_model_loss(params_and_buffers, data, target):
            """Compute the loss of a set of params on a batch of data, via the meta model"""
            predictions = functional_call(meta_model, params_and_buffers, (data,))
            predictions = predictions.double() # This has a significant effect!
            return criterion(predictions, target)

        # define a loss function that takes the stacked ensemble params, data, and target
        # in_dims=(0, None, None) adds an ensemble dimension to the first argument 'params_and_buffers' but not to the other two arguments
        vmap_loss = vmap(meta_model_loss, in_dims=(0, None, None))
        
        batch_losses = []  # List to store batch losses

        with profiler(f'Evaluating stacked ensemble on {device}'):
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                batch_loss = vmap_loss(stacked_ensemble, data, target)
                batch_losses.append(batch_loss)
        
        with profiler(f'Moving batch loss to cpu'):
            stacked_losses = torch.stack(batch_losses).cpu().numpy()

        with profiler('Freeing data and target'):
            del data, target
            torch.cuda.empty_cache()

        with profiler('Freeing stacked ensemble'):
            del stacked_ensemble
            torch.cuda.empty_cache()
        
        # Summing in numpy for the more precise pairwise summation algorithm.
        average_losses = stacked_losses.mean(axis=0)
        assert average_losses.dtype == np.float64, "Ensure that the loss is calculated in double precision"
        
        return average_losses

ensemble_loss = eval_ensemble(ensemble_list, dataloader, criterion, device)

# %%
# Convert the loss to a list and unpack
center_loss, *dir_losses = ensemble_loss.tolist()

# Ensure the length of dir_losses matches the DataFrame length
assert len(dir_losses) == len(df)

# Add the directional losses to the DataFrame
df['Loss'] = dir_losses

for direction in df.index.get_level_values('Direction'):
    df.loc[(direction, 0), ['Distance', 'Loss']] = [0., center_loss]

df.sort_index(inplace=True)

# %%
df['Grit'] = np.nan

def grit(dist: np.ndarray, loss: np.ndarray) -> np.ndarray:
    """
    Calculate the roughness of a loss function at different scales.
    """
    assert np.all(2 == dist[2:]/dist[1:-1]), "Each distance must be twice the previous distance"
    
    return (loss[0] - 2*loss[1:-1] + loss[2:]) / dist[1:-1]

# Calculate the grit, one direction at a time
for direction, group in df.groupby(level='Direction'):
    dist = group['Distance'].to_numpy()
    loss = group['Loss'].to_numpy()

    # Calculate the grit values and pad with NaNs
    grit_values = [np.nan] + list(grit(dist, loss)) + [np.nan]

    # Update the DataFrame
    df.loc[(direction,), 'Grit'] = grit_values

print(df)

import plotly.graph_objects as go
from typing import List

def plot(df: pd.DataFrame, description: str) -> List[go.Figure]:

    # Common layout settings
    common_layout = {
        'legend_title': 'Direction',
        'template': 'seaborn',
        'height': 600
    }

    profile_fig = go.Figure(layout={
        **common_layout,
        'title': f'Loss Landscape Profiles of {description}',
        'xaxis': {'title': 'Distance from Center in Parameter Space'},
        'yaxis': {'title': 'Loss'}
    })
    
    grit_fig = go.Figure(layout={
        **common_layout,
        'title': f'Grit of {description}',
        'xaxis': {'type': 'log', 'title': 'Coarse Graining Scale', 'tickformat': '.0e'},
        'yaxis': {'title': 'Grit'}
    })
    
    abs_grit_fig = go.Figure(layout={
        **common_layout,
        'title': f'Grit Magnitude of {description}',
        'xaxis': {'type': 'log', 'title': 'Coarse Graining Scale', 'tickformat': '.0e'},
        'yaxis': {'type': 'log', 'title': '|Grit|', 'tickformat': '.0e'}
    })

    # Build all plots within one loop
    grouped_data = df.groupby(level='Direction')
    for direction, data in grouped_data:
        # Profile plot
        profile_fig.add_trace(go.Scatter(
            y=data['Loss'],
            x=data['Distance'],
            mode='lines+markers',
            name=direction
        ))
    
        # Grit plot
        grit_fig.add_trace(go.Scatter(
            x=data['Distance'],
            y=data['Grit'],
            mode='lines+markers',
            name=direction
        ))

        # Grit magnitude plot
        abs_grit_fig.add_trace(go.Scatter(
            x=data['Distance'],
            y=np.abs(data['Grit']),
            mode='lines+markers',
            name=direction
        ))

    return [profile_fig, grit_fig, abs_grit_fig]

figs = plot(df, 'Simple MLP on MNIST')
for fig in figs:
    fig.show()