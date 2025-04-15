import torch

# Path to the checkpoint
ckpt_path = "log/latent_diffusion/audioldm_original_with_imagebind/audioldm_with_imagebind/checkpoints/checkpoint-fad-133.00-global_step=244999.ckpt"

# Load the checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")

# Print all top-level keys
print("Top-level keys in checkpoint:")
for key in ckpt.keys():
    print(f"  {key}")

# Explore the state_dict keys (model parameters)
# state_dict = ckpt["state_dict"]
# print("\nModel parameter keys in state_dict:")
# for i, key in enumerate(state_dict.keys()):
#     print(f"{i+1:03d}: {key}")

# Optionally: Print shape of specific parameters
# print("\nParameter shapes:")
# for name, param in state_dict.items():
#     if hasattr(param, 'shape'):
#         print(f"{name}: {param.shape}")
