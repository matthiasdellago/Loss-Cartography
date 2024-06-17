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
from tqdm import tqdm
# custom modules
from utils.profiler import profiler
from utils.param_math import iadd, scale, sub, norm, normalize, rand_like, project_to_module

# TODO: make funcs importable without running the whole script. Seperate experiment template/demo and functions

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

torch.set_grad_enabled(False)
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

# config dict for the experiment
cfg = {
    'device': DEVICE,
    'precision': torch.float32,          # float32 or float64
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
    'grad': False,                        # should we look in the direction of gradient ascent and descent?
    'subspaces': ['fc1', 'weight', 'bias'], # project directions onto modules with these substrings in their names. See project_to_module()
    'rand_dirs': 20 if CUDA else 0,     # number of random directions to add, in addition to the gradient and radial directions
    'max_oom':    2 if CUDA else 0,     # furthest sample will be 10**max_oom from the center
    'min_oom':  -13 if CUDA else -1,    # closest sample will be 10**min_oom from the center
}

torch.set_default_dtype(cfg['precision']) # precision global default
# SUPER IMPORTANT
# float64:  - Finite differences noise on the oom of 1e-15.
#           - Transition from 'Quadratic' to 'Noise' behaviour at parameter distances on the order of 1e-6
#           - Example takes 350 seconds
#           - ca. 9 GB GPU RAM
# float32:  - Finite differences noise on the oom of 1e-8.
#           - Transition from 'Quadratic' to 'Noise' behaviour at parameter distances on the order of 1e-3
#           - Example takes 35 seconds
#           - ca. 4.5 GB GPU RAM

print(f'Running on {cfg["device"]}')

def directions(c:dict) -> dict:
    """
    Contains:
    - Radially Out: The direction away from the center
    - Radially In: The direction towards the center
    - Random i: Random directions
    if config['grad'] is True:
        - Ascent: The direction of gradient ascent
        - Descent: The direction of gradient descent
    for subspace in config['subspaces']:
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
        
        with torch.set_grad_enabled(True):
            with profiler('1 step of gradient descent over the whole dataset'):
                for data, target in tqdm(dataloader):
                    data, target = data.to(device), target.to(device)
                    output = grad_model(data)
                    loss = criterion(output, target)
                    loss.backward()

        # we want the direction of ascent, as a model:
        for param in grad_model.parameters():
            param.data = param.grad.data

        return normalize(grad_model)

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
    assert all(torch.isclose(norm(d), torch.tensor(1.)) for d in dirs.values())

    return dirs

dirs = directions(cfg)

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

df = dirs_and_dists(dirs.keys(), cfg['min_oom'], cfg['max_oom'])

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

df = dists_to_models(df, dirs, cfg['center'])

# add the center model
ensemble_list = [cfg['center']] + list(df['Model'])
df.drop(columns='Model', inplace=True)

@torch.no_grad
def eval_ensemble(ensemble_list:[nn.Module], dataloader:DataLoader, criterion:nn.functional, device:torch.device) -> torch.Tensor:
    """Parallel evaluation with vmap()"""
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
            predictions = predictions.double() # This has a significant effect! If not torch.set_default_dtype(torch.float64) anyway
            return criterion(predictions, target)

        # define a loss function that takes the stacked ensemble params, data, and target
        # in_dims=(0, None, None) adds an ensemble dimension to the first argument 'params_and_buffers' but not to the other two arguments
        vmap_loss = vmap(meta_model_loss, in_dims=(0, None, None))
        
        batch_losses = []  # List to store batch losses

        with profiler(f'Evaluating stacked ensemble om {device}'):
            for data, target in tqdm(dataloader):
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
        assert average_losses.dtype == np.float64, "Ensure that the loss was calculated in double precision"
        
        return average_losses

ensemble_loss = eval_ensemble(ensemble_list, cfg['dataloader'], cfg['criterion'], cfg['device'])

# Convert the loss to a list and unpack
center_loss, *df['Loss'] = ensemble_loss.tolist()

# Add the center loss to the DataFrame at 'Distance' = 0
for direction in df.index.get_level_values('Direction'):
    df.loc[(direction, 0), ['Distance', 'Loss']] = [0., center_loss]

df.sort_index(inplace=True)

def curvature_scale_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the curvature, grit, and finite difference as a function of the scale of the coarse graining.
    Add columns for 'Curvature', 'Grit', and 'Finite Difference'.
    """
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
    
    return df

df = curvature_scale_analysis(df)

df.head()

from utils.plots import plot_df, save_fig_with_cfg

figs = plot_df(df, 'Simple MLP on MNIST')

for fig in figs:
    fig.show(config={
        'toImageButtonOptions': {
            'format': 'png',
            'filename': fig.layout.title.text,
            'height': 600,
            'width': 800,
            'scale': 1
        }
    })
    # Save the figures
    save_fig_with_cfg(dir='automatic_figs',fig=fig, config=cfg)

