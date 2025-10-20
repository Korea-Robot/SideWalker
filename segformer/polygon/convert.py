import torch
from safetensors.torch import save_file

# --- CONFIGURE THIS ---
OLD_MODEL_PATH = "best_seg_model.pth"
NEW_MODEL_PATH = "best_seg_model.safetensors"

# Load the state dictionary from your old .pth file
weights = torch.load(OLD_MODEL_PATH, map_location="cuda")

# Save the weights in the new, safe format
save_file(weights, NEW_MODEL_PATH)

print(f"Model successfully converted from {OLD_MODEL_PATH} to {NEW_MODEL_PATH}")
