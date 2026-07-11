#!/usr/bin/env python
"""Test just the frame extraction part"""

import os
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.utils_preprocess import extract_video_frames_to_disk

def test_frame_extraction():
    video_path = "data/otowa_koregaseikai.mov"
    
    # Create temporary output directory
    temp_dir = tempfile.mkdtemp(prefix="test_frames_")
    
    try:
        print(f"Testing frame extraction from: {video_path}")
        print(f"Output directory: {temp_dir}")
        
        # Extract frames at default rate (every frame)
        frame_paths = extract_video_frames_to_disk(
            video_input=video_path,
            output_dir=temp_dir,
            k=1,  # Every frame
            max_size=1024
        )
        
        print(f"\nExtracted {len(frame_paths)} frames")
        
        # Test sampling at 3fps (assuming 30fps source)
        if len(frame_paths) > 0:
            # Simulate 3fps extraction
            step = max(1, 30 // 3)  # Every 10th frame for 30fps -> 3fps
            sampled = frame_paths[::step]
            print(f"3fps sampling would give: {len(sampled)} frames")
            
            # Show first few frame names
            print("\nFirst 5 frames:")
            for i, path in enumerate(frame_paths[:5]):
                print(f"  {os.path.basename(path)}")
                
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up: {temp_dir}")

if __name__ == "__main__":
    test_frame_extraction()