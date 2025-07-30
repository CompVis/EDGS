#!/usr/bin/env python
"""Test script for COLMAP reconstruction with automatic_reconstructor-like settings"""

import os
import sys
import time

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from source.utils_preprocess import orchestrate_video_to_colmap_scene

def main():
    video_path = "data/otowa_koregaseikai.mov"
    outputs_dir = "outputs/otowa_automatic_test"
    
    print(f"Starting COLMAP reconstruction test for: {video_path}")
    print("Using automatic_reconstructor-like settings...")
    
    start_time = time.time()
    
    # Run the orchestration with automatic mode
    _, scene_dir = orchestrate_video_to_colmap_scene(
        video_path,
        num_ref_views=10,  # This will be ignored in automatic mode
        max_size=1024,
        base_work_dir=outputs_dir,
        use_automatic_mode=True
    )
    
    end_time = time.time()
    
    if scene_dir:
        print(f"\n✅ COLMAP reconstruction completed successfully!")
        print(f"📁 Scene directory: {scene_dir}")
        print(f"⏱️  Total time: {(end_time - start_time)/60:.1f} minutes")
        
        # Check reconstruction results
        sparse_dir = os.path.join(scene_dir, "sparse", "0")
        if os.path.exists(sparse_dir):
            files = os.listdir(sparse_dir)
            print(f"\n📊 Reconstruction files: {files}")
            
            # Try to load and print stats
            try:
                import pycolmap
                reconstruction = pycolmap.Reconstruction(sparse_dir)
                print(f"\n📈 Reconstruction statistics:")
                print(f"   - Cameras: {len(reconstruction.cameras)}")
                print(f"   - Images: {len(reconstruction.images)}")
                print(f"   - 3D Points: {len(reconstruction.points3D)}")
            except:
                print("\n⚠️  Could not load reconstruction statistics")
    else:
        print("\n❌ COLMAP reconstruction failed!")

if __name__ == "__main__":
    main()