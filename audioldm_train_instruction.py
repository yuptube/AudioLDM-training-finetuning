import subprocess

# Run the Python command for training with subprocess
command = [
    "python3", 
    "audioldm_train/train/latent_diffusion.py", 
    "-c", "audioldm_train/config/audioldm_original_with_rdm/audioldm_with_rdm.yaml",
    "--reload_from_ckpt", "data/checkpoints/audioldm-s-full"
]

# Run the command and capture the output
process = subprocess.run(command, capture_output=True, text=True)

# Print the output and error (if any) for debugging

print(process.stdout)

print(process.stderr)

# Optionally, check if the process was successful
if process.returncode == 0:
    print("Training completed successfully!")
else:
    print(f"Training failed with exit code {process.returncode}.")
