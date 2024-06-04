#!/bin/bash
# Command line tool to execute a Jupyter notebook, and save the output back to the notebook.
# Usage: ./nbx.sh <notebook_name.ipynb>

if [ -z "$1" ]; then
    echo "Error: No notebook name provided."
    echo "Usage: $0 <notebook_name.ipynb>"
    exit 1
fi

NOTEBOOK="$1"

jupyter nbconvert --execute --to notebook --inplace "$NOTEBOOK"