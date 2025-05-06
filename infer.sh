#!/bin/bash
#SBATCH --job-name=audioldm_train
#SBATCH --output=cluster_logs/%x_%j_infer.out       # Standard output log
#SBATCH --error=cluster_logs/%x_%j_infer.err        # Standard error log
#SBATCH --partition=a100            # Example for 2080ti partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --time=00:03:00
#SBATCH --mail-type=END,FAIL        
#SBATCH --mail-user=ys01085@surrey.ac.uk

# Set up Conda
export PATH="$HOME/miniconda3/condabin:$PATH"  
eval "$($HOME/miniconda3/condabin/conda shell.bash hook)"  
# Add 'audioldm_train' directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/mnt/fast/nobackup/users/ys01085/AudioLDM-training-finetuning

# Activate Conda environment
if conda info --envs | grep -q "audioldm_train"; then
    echo "✅ Activating environment 'audioldm_train'..."
    conda activate audioldm_train
else
    echo "❌ Conda environment 'audioldm_train' not found. Exiting..."
    exit 1
fi

set -e  # Exit on any error
trap 'echo "⚠️ An error occurred while running the Python script. Exiting..." >&2; exit 1' ERR

python3 imagebind_to_audio.py
echo "✅ Training script completed successfully."