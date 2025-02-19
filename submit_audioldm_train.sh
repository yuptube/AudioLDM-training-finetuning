#!/bin/bash
#SBATCH --job-name=audoldm_train
#SBATCH --output=cluster_logs/%x_%j.out       # Standard output log
#SBATCH --error=cluster_logs/%x_%j.err        # Standard error log
#SBATCH --partition=2080ti  # Example for 2080ti partition
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL        
#SBATCH --mail-user=ys01085@surrey.ac.uk

# Set up the path to your personal Conda installation
export PATH="$HOME/miniconda3/condabin:$PATH"  # Add Conda to your path
echo "Current working directory: $(pwd)"
# Initialize Conda for your shell
eval "$($HOME/miniconda3/condabin/conda shell.bash hook)"  # Initialize Conda
# Add 'audioldm_train' directory to PYTHONPATH to make sure Python can find the module
export PYTHONPATH=$PYTHONPATH:/mnt/fast/nobackup/users/ys01085/AudioLDM-training-finetuning

# Check Python and PYTHONPATH
echo "Using Python from: $(which python)" 
echo "PYTHONPATH: $PYTHONPATH" 


# Check if the environment is activated successfully
if conda info --envs | grep -q "audioldm_train"; then
    echo "Environment 'audioldm_train' exists."
    # Activate the conda environment
    conda activate audioldm_train
else
    echo "Environment 'audioldm_train' does not exist."
    exit 1  # Exit the script if the environment is not found
fi


# set -e  # Exit on any error

# # Set the trap for errors
# trap 'echo "An error occurred while running the Python script. Exiting..." >&2; exit 1' ERR

# # Try block
# echo "Running the Python script..."
# python3 audioldm_train_instruction.py  # If this fails, the trap will catch it

# # If successful
# echo "Python script ran successfully."