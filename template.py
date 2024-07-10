#template.py
from utils.core_logic import directions, dists_to_models, dirs_and_dists, eval_ensemble, curvature_scale_analysis
from utils.plots import plot_df, save_fig_with_cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
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
        pin_memory=CUDA, # This is super important for speed!
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

dirs = directions(cfg)

df = dirs_and_dists(dirs.keys(), cfg['min_oom'], cfg['max_oom'])

df = dists_to_models(df, dirs, cfg['center'])

# add the center model
ensemble_list = [cfg['center']] + list(df['Model'])
df.drop(columns='Model', inplace=True)

ensemble_loss = eval_ensemble(ensemble_list, cfg['dataloader'], cfg['criterion'], cfg['device'])

# Convert the loss to a list and unpack
center_loss, *df['Loss'] = ensemble_loss.tolist()

# Add the center loss to the DataFrame at 'Distance' = 0
for direction in df.index.get_level_values('Direction'):
    df.loc[(direction, 0), ['Distance', 'Loss']] = [0., center_loss]

df.sort_index(inplace=True)

df = curvature_scale_analysis(df)

df.head()

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