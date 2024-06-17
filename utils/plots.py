import xml.etree.ElementTree as ET
from xml.dom import minidom
import json
import os
from typing import List
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_df(df: pd.DataFrame, description: str) -> List[go.Figure]:
    """
    Takes a DataFrame with columns ['Loss', 'Distance', 'Finite Difference', 'Grit', 'Curvature']
    and a desciption of the model, and returns a list of plotly figures.
    """
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
            'title': f'abs({name}) of {description}',
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

def save_fig_with_cfg(dir:str, fig:go.Figure, config:dict) -> None:
    """
    Save a plotly figure as an SVG file in dir, and embed the configuration as metadata.
    """
    # Configure filename from plot title and directory
    filename = f"{fig.layout.title.text.replace(' ', '_')}.svg"
    filename = os.path.join(dir, filename)
    # Save the figure as an SVG file
    fig.write_image(filename)
    
    # Parse the saved SVG file and prepare metadata element
    tree = ET.parse(filename)
    root = tree.getroot()
    metadata = ET.Element("metadata")
    metadata.text = json.dumps(config, indent=4, default=str)
    
    # Insert metadata as the first child of the root element
    root.insert(0, metadata)
    
    # Generate formatted XML string and save it
    pretty_xml = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(filename, "w") as file:
        file.write(pretty_xml)