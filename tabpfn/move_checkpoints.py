import os
import re
import shutil

source_dir = "model_checkpoints"
target_dir = "/data/parietal/store/work/lgrinszt/TabPFN/tabpfn/model_checkpoints"

# Get a list of all checkpoint files
checkpoint_files = [f for f in os.listdir(source_dir) if f.endswith(".pt")]

# Create a dictionary to store the latest checkpoint for each unique checkpoint name
latest_checkpoints = {}

# Iterate through the checkpoint files
for checkpoint_file in checkpoint_files:
    match = re.match(r"(.*)_(\d+).pt", checkpoint_file)
    if match:
        checkpoint_name, epochs = match.groups()
        epochs = int(epochs)

        # Update the dictionary with the latest checkpoint
        if checkpoint_name not in latest_checkpoints or epochs > latest_checkpoints[checkpoint_name][1]:
            latest_checkpoints[checkpoint_name] = (checkpoint_file, epochs)

# Create the target directory if it does not exist
os.makedirs(target_dir, exist_ok=True)

# Copy the latest checkpoints to the target directory
for checkpoint_name, (checkpoint_file, _) in latest_checkpoints.items():
    source_path = os.path.join(source_dir, checkpoint_file)
    target_path = os.path.join(target_dir, checkpoint_file)
    shutil.copyfile(source_path, target_path)
    print(f"Copied {checkpoint_file} to {target_path}")
