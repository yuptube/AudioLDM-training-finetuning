#!/bin/bash
#SBATCH --job-name=audioldm_train
#SBATCH --output=cluster_logs/%x_%j.out       # Standard output log
#SBATCH --error=cluster_logs/%x_%j.err        # Standard error log
#SBATCH --partition=debug                 # Example for 2080ti partition
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=25G
#SBATCH --time=0:15:00
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
# $LATEST_CHECKPOINT=""
# $CHECKPOINT_DIR=""
# if [ -z "$LATEST_CHECKPOINT" ]; then
# echo "No checkpoint found, starting fresh... (form audioldm -s )"
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/audioldm_original_with_imagebind/audioldm_with_imagebind.yaml \
    --reload_from_ckpt data/checkpoints/audioldm-s-full.ckpt
# else
#     echo "Resuming from checkpoint: $LATEST_CHECKPOINT"
#     python3 latent_diffusion_slrum.py -c audioldm_train/config/audioldm_original_with_rdm/audioldm_with_rdm.yaml --reload_from_ckpt $CHECKPOINT_DIR/$LATEST_CHECKPOINT
# fi

echo "✅ Training script completed successfully."