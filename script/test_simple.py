#!/usr/bin/env python
# Simple test to verify the directory creation logic without full dependencies

import os
import argparse

# Simulate the argument parsing from fit_model_to_scene_full.py
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, default="test.mp4")
parser.add_argument("--output_dir", type=str, default=None)
args = parser.parse_args()

print(f"video_path: {args.video_path}")
print(f"output_dir: {args.output_dir}")

# Simulate the directory logic from the script
scene_dir = "/fake/scene/dir"  # This would normally come from COLMAP processing

# Determine output directory for EDGS training results
if args.output_dir:
    # Use specified output directory
    model_path = args.output_dir
    print(f"Using custom output directory: {model_path}")
else:
    # Default behavior: create models subfolder in scene directory
    model_path = os.path.join(scene_dir, "models")
    print(f"Using default models directory: {model_path}")

print(f"Creating directory: {model_path}")
os.makedirs(model_path, exist_ok=True)
print(f"Directory created successfully: {os.path.exists(model_path)}")

# Create a test file to verify the directory works
test_file = os.path.join(model_path, "test.txt")
with open(test_file, "w") as f:
    f.write("Test output directory creation")
    
print(f"Test file created: {test_file}")
print("SUCCESS: Output directory creation works correctly!")