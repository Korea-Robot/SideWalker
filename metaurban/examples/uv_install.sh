#!/bin/bash

# This script installs the MetaUrban driving simulator using the 'uv'
# package manager and virtual environment tool.

# --- Step 1: Clone the Repository ---
# This command downloads the source code from GitHub.
echo "Cloning the MetaUrban repository..."
git clone https://github.com/metadriverse/metaurban.git
cd metaurban

# --- Step 2: Create and Activate Virtual Environment ---
# We use 'uv venv' to create a new virtual environment named '.venv'
# specifying Python 3.9. This is equivalent to 'conda create'.
echo "Creating virtual environment with Python 3.9 using uv..."
uv venv -p 3.9

# Activate the environment. The command depends on your operating system.
# On macOS and Linux:
echo "Activating the virtual environment..."
source .venv/bin/activate
# On Windows (in Command Prompt or PowerShell):
# .venv\Scripts\activate

# --- Step 3: Install MetaUrban ---
# This installs the MetaUrban package in "editable" mode, which is useful
# for development. 'uv pip' is used here for compatibility with pip commands.
echo "Installing MetaUrban in editable mode..."
uv pip install -e .

# --- Step 4: Install and Compile ORCA Module ---
# ORCA is used for pedestrian trajectory generation.
# First, install its dependency, pybind11.
echo "Installing pybind11 for ORCA module..."
uv pip install pybind11

# Navigate to the ORCA directory, clean previous builds, and compile.
echo "Compiling the ORCA algorithm module..."
cd metaurban/orca_algo && rm -rf build
bash compile.sh && cd ../..

# --- Step 5: Install Learning and Visualization Libraries ---
# These packages are required for reinforcement learning, imitation learning,
# and visualizing results with tools like TensorBoard and WandB.
echo "Installing RL, IL, and visualization requirements..."
uv pip install stable_baselines3 imitation tensorboard wandb scikit-image pyyaml gdown

echo "Installation complete! The 'metaurban' environment is ready."

# uv cache clean