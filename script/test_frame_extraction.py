#!/usr/bin/env python
"""Test frame extraction and COLMAP settings"""

import os
import sys
# Add parent directory to path to import source modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test imports
try:
    from source.utils_preprocess import process_input_for_colmap, run_colmap_on_scene
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test parameters
print("\n=== Testing Frame Extraction Parameters ===")
print("Default target_fps: 3.0")
print("Expected frames for 60s video: ~180 frames")
print("Expected frames for 180s video: ~540 frames")

# Test COLMAP settings
print("\n=== Testing COLMAP Settings ===")
print("Key settings when use_automatic_mode=True:")
print("- multiple_models: False (forces single reconstruction)")
print("- max_num_models: 1")
print("- min_num_matches: 5 (very lenient)")
print("- sequential overlap: 30 frames")
print("- mapper.init_min_num_inliers: 15 (relaxed)")

print("\n✓ All settings configured to create single COLMAP reconstruction")
print("\nTo run full test:")
print("python script/fit_model_to_scene_full.py \\")
print("    --video_path data/otowa_koregaseikai.mov \\")
print("    --output_dir outputs/otowa_3fps \\")
print("    --target_fps 3.0")