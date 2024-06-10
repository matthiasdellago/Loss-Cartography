# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from typing import List
from torch.func import stack_module_state, functional_call
from torch import vmap
from copy import deepcopy
# custom modules
from utils.profiler import profiler

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

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

# config dict for the experiment
c = {
    'device': DEVICE,
    'center': SimpleMLP().to(DEVICE),   # the point in parameter space to start from
    'dataloader': DataLoader(
        MNIST('./data', download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=100,
        num_workers=os.cpu_count(),
        pin_memory=CUDA,
    ),
    'criterion': F.cross_entropy,
    'grad': True,                        # should we look in the direction of gradient ascent and descent?
    'subspaces': ['fc1', 'weight', 'bias'], # project directions onto modules with these substrings in their names. See project_to_module()
    'rand_dirs': 20 if CUDA else 0,     # number of random directions to add, in addition to the gradient and radial directions
    'max_oom':    2 if CUDA else 0,     # furthest sample will be 10**max_oom from the center
    'min_oom':  -13 if CUDA else -1,    # closest sample will be 10**min_oom from the center
}

# if grad is False, we don't need gradients
torch.set_grad_enabled(c['grad'])

print(f'Running on {c["device"]}')

# %% 
# Define utility functions for parameter space arithmetic. iadd is inplace, all others create a new model.

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
def directions(c:dict) -> dict:
    """
    Dictionary of directions in parameter space, keyes = names of directions, values = normalized models.

    Contains:
    - Radially Out: The direction away from the center
    - Radially In: The direction towards the center
    for i in range(config['rand_dirs']):
        - Random i: Random directions
    if config['grad'] is True:
        - Ascent: The direction of gradient ascent
        - Descent: The direction of gradient descent
    for subspace in config['subspaces']:
        for direction in directions:
            - direction ⋅ P(subspace): The projection of each direction onto the subspace
    """
    radial = normalize(c['center'])

    dirs = {
        'Radially Out': radial,
        'Radially In': scale(radial,-1),
        }

    # add random directions
    dirs.update({f'Random {i}': rand_like(radial) for i in range(c['rand_dirs'])})

    def ascent_direction(model:nn.Module, criterion:nn.functional, dataloader:DataLoader, device:torch.device) -> nn.Module:
        """Find the direction of gradient ascent for a given model, and return it as a normalized model."""
        # create a model to find the gradient
        grad_model = deepcopy(model).to(device)
        optimizer = torch.optim.SGD(grad_model.parameters(), lr=1)
        
        torch.set_grad_enabled(True)
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
        # grad_model = grad_model.cpu()
        dir_ascent = sub(model, grad_model) # from updated model to original model, back up the gradient
        return normalize(dir_ascent)

    if c['grad']:
        dir_ascent = ascent_direction(c['center'], c['criterion'], c['dataloader'], c['device'])
        dir_descent = scale(dir_ascent, -1) # descent = -1*ascent
        dirs.update({'Ascent': dir_ascent,'Descent': dir_descent})

    # add projections of each dir in dirs onto each SUBSPACE
    for dir_name, direction in list(dirs.items()):
        for subspace in c['subspaces']:
            projected_dir = project_to_module(direction, subspace)
            new_key = f'{dir_name} ⋅ P({subspace})'
            dirs[new_key] = projected_dir

    # check that they are all normalized
    assert all(torch.isclose(abs(d), torch.tensor(1.0)) for d in dirs.values())

    return dirs

dirs = directions(c)

# %%

def dirs_and_dists(dir_names:[str], MIN_OOM:int, MAX_OOM:int) -> pd.DataFrame:
    """
    Create a DataFrame with a MultiIndex of direction names and doubling steps,
    containing logarithmically spaced 'Distance' values from 10^MIN_OOM to just over 10^MAX_OOM.
    """
    # How often do I need to double 10^MIN_OOM to reach 10^MAX_OOM?
    doublings = int(np.ceil((MAX_OOM - MIN_OOM) * np.log2(10)))
    steps = np.arange(doublings, dtype=np.float64)

    # Compute all the distances we need to sample.
    dists = 2**steps * 10**MIN_OOM
    
    # In order to sample the spaces between 2^scale and 2^(scale+1),
    # and uniformly sample the log-space, we multiply each direction by a different offset
    offset_per_dir = np.logspace(base=2, start=0, stop=1, num=len(dir_names), endpoint=False, dtype=np.float64)

    # Compute the outer product of all offsets and distances.
    dists_per_dir = np.outer(offset_per_dir, dists)

    # Create an 'outer product' of the corresponding indeces using pd.MultiIndex.from_product
    multi_index = pd.MultiIndex.from_product([dir_names, steps+1], names=['Direction', 'Step'])
    
    # Create DataFrame directly with the computed distances.
    return pd.DataFrame({'Distance': dists_per_dir.flatten()}, index=multi_index)

df = dirs_and_dists(dirs.keys(), c['min_oom'], c['max_oom'])

# %%
@torch.no_grad
def dists_to_models(df:pd.DataFrame, dirs:dict, center:nn.Module) -> pd.DataFrame:
    """
    Fill the DataFrame with the models corresponding to the distances in the 'Distance' column.
    """
    for (dir_name, step), distance in df['Distance'].items():
        direction = dirs[dir_name]
        shift = scale(direction,distance) # shift = direction * distance
        location = iadd(shift, center)    # location = center + shift
        df.at[(dir_name, step), 'Model'] = location
    return df

df = dists_to_models(df, dirs, c['center'])

# %%
# Parallel evaluation with vmap()

# add the center model
ensemble_list = [c['center']] + list(df['Model'])
df.drop(columns='Model', inplace=True)

@torch.no_grad
def eval_ensemble(ensemble_list:[nn.Module], dataloader:DataLoader, criterion:nn.functional, device:torch.device) -> torch.Tensor:

    with profiler(f'Ensemble size: {len(ensemble_list)}', pad_char='-'):

        ensemble_size = len(ensemble_list)
        
        # stack to prepare for vmap
        with profiler('Stacking Ensemble'):
          stacked_ensemble = stack_module_state(ensemble_list)

        # Construct a "stateless" version of one of the models. It is "stateless" in
        # the sense that the parameters are meta Tensors and do not have storage.
        meta_model = deepcopy(ensemble_list[0]).to('meta')

        with profiler('Deleting Ensemble List'):
          del ensemble_list

        def meta_model_loss(params_and_buffers, data, target):
            """Compute the loss of a set of params on a batch of data, via the meta model"""
            predictions = functional_call(meta_model, params_and_buffers, (data,))
            predictions = predictions.double() # This has a significant effect!
            return criterion(predictions, target)

        # define a loss function that takes the stacked ensemble params, data, and target
        # in_dims=(0, None, None) adds an ensemble dimension to the first argument 'params_and_buffers' but not to the other two arguments
        vmap_loss = vmap(meta_model_loss, in_dims=(0, None, None))
        
        batch_losses = []  # List to store batch losses

        with profiler(f'Evaluating stacked ensemble om {device}'):
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

ensemble_loss = eval_ensemble(ensemble_list, c['dataloader'], c['criterion'], c['device'])

# %%
# Convert the loss to a list and unpack
center_loss, *dir_losses = ensemble_loss.tolist()

# Ensure the length of dir_losses matches the DataFrame length
assert len(dir_losses) == len(df)

# Add the directional losses to the DataFrame
df['Loss'] = dir_losses

# Add the center loss to the DataFrame at 'Distance' = 0
for direction in df.index.get_level_values('Direction'):
    df.loc[(direction, 0), ['Distance', 'Loss']] = [0., center_loss]

df.sort_index(inplace=True)

# %%
def finite_difference(dist: np.ndarray, loss: np.ndarray) -> np.ndarray:
    """
    Calculate the finite difference of 3 equally spaced points.
    """   
    return loss[0] - 2*loss[1:-1] + loss[2:]

# Calculate the grit, one direction at a time
for direction, group in df.groupby(level='Direction'):
    dist = group['Distance'].to_numpy()
    loss = group['Loss'].to_numpy()

    assert dist.dtype == np.float64
    assert loss.dtype == np.float64

    finite_diff = finite_difference(dist, loss)
    df.loc[(direction,), 'Finite Difference'] = np.concatenate([[np.nan], finite_diff, [np.nan]])

    grit = finite_diff / dist[1:-1]
    df.loc[(direction,), 'Grit'] = np.concatenate([[np.nan], grit, [np.nan]])

    curvature = finite_diff / dist[1:-1]**2
    df.loc[(direction,), 'Curvature'] = np.concatenate([[np.nan], curvature, [np.nan]])

df.head()

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

    finit_diff_figs = {"Finite Difference": go.Figure(), "Grit": go.Figure(), "Curvature": go.Figure()}

    for name, fig in finit_diff_figs.items():
        fig.update_layout({
            **common_layout,
            'title': f'|{name}| of {description}',
            'xaxis': {'type': 'log', 'title': 'Coarse Graining Scale', 'tickformat': '.0e'},
            'yaxis': {'type': 'log', 'title': f'|{name}|', 'tickformat': '.0e'},
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

        # Finite Difference plots
        for name,fig in finit_diff_figs.items():
            fig.add_trace(go.Scatter(
                x=data['Distance'],
                y=np.abs(data[name]),
                mode='lines+markers',
                name=direction
            ))

    return [profile_fig] + list(finit_diff_figs.values())

figs = plot(df, 'Simple MLP on MNIST')
for fig in figs:
    fig.show()