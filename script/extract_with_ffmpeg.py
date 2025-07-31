#!/usr/bin/env python
"""Extract video frames using ffmpeg directly"""

import subprocess
import os

def extract_frames_ffmpeg(video_path, output_dir, target_fps=3.0, max_size=1024):
    """Extract frames using ffmpeg directly"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video info using ffprobe
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,duration',
            '-of', 'csv=p=0', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = result.stdout.strip().split(',')
        width, height, frame_rate, duration = info[0], info[1], info[2], info[3]
        
        print(f"Video info: {width}x{height}, frame_rate: {frame_rate}, duration: {duration}s")
        
        # Calculate how many frames to extract
        if duration and float(duration) > 0:
            total_duration = float(duration)
            target_frames = int(total_duration * target_fps)
            print(f"Will extract ~{target_frames} frames at {target_fps} fps")
        else:
            target_frames = 300  # Default fallback
            
    except Exception as e:
        print(f"Could not get video info: {e}")
        target_frames = 300
    
    # Extract frames using ffmpeg
    try:
        # Scale down if needed and extract at target fps
        scale_filter = f"scale='min({max_size},iw)':'min({max_size},ih)':force_original_aspect_ratio=decrease"
        fps_filter = f"fps={target_fps}"
        
        cmd = [
            'ffmpeg', '-i', video_path, '-y',
            '-vf', f"{fps_filter},{scale_filter}",
            '-q:v', '2',  # High quality
            '-frames:v', str(target_frames),  # Limit number of frames
            os.path.join(output_dir, 'frame_%08d.jpg')
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Count extracted frames
        frame_files = [f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.jpg')]
        print(f"Successfully extracted {len(frame_files)} frames")
        
        return sorted([os.path.join(output_dir, f) for f in frame_files])
        
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg extraction failed: {e}")
        print(f"stderr: {e.stderr}")
        return []

if __name__ == "__main__":
    video_path = "data/tower_by_drone.MP4"
    output_dir = "outputs/test_frames"
    
    frames = extract_frames_ffmpeg(video_path, output_dir, target_fps=2.0, max_size=512)
    print(f"Extracted {len(frames)} frames to {output_dir}")
    
    if frames:
        print("First few frames:")
        for frame in frames[:5]:
            print(f"  {frame}")
    else:
        print("No frames extracted!")