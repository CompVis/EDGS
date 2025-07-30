# This file contains function for video or image collection preprocessing.
# For video we do the preprocessing and select k sharpest frames.
# Afterwards scene is constructed
import os
import time

import cv2
import numpy as np
import pycolmap
from matplotlib import pyplot as plt
from moviepy import VideoFileClip
from PIL import Image
from tqdm import tqdm


def get_rotation_moviepy(video_path):
    clip = VideoFileClip(video_path)
    rotation = 0

    try:
        displaymatrix = clip.reader.infos["inputs"][0]["streams"][2]["metadata"].get(
            "displaymatrix", ""
        )
        if "rotation of" in displaymatrix:
            angle = float(
                displaymatrix.strip().split("rotation of")[-1].split("degrees")[0]
            )
            rotation = int(angle) % 360

    except Exception as e:
        print(f"No displaymatrix rotation found: {e}")

    clip.reader.close()
    # if clip.audio:
    #    clip.audio.reader.close_proc()

    return rotation


def resize_max_side(frame, max_size):
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def extract_video_frames_to_disk(video_input, output_dir, k=1, max_size=1024, use_all_frames=False):
    """
    Extracts every k-th frame from a video using ffmpeg, saves to disk to avoid memory overflow.

    Parameters:
        video_input (str, file-like, or list): Path to video file, file-like object, or list of image files.
        output_dir (str): Directory to save extracted frames.
        k (int): Interval for frame extraction (every k-th frame).
        max_size (int): Maximum size for width or height after resizing.

    Returns:
        frame_paths (list): List of paths to extracted frame files.
    """
    import subprocess
    import tempfile
    import shutil
    
    # Handle list of image files (not single video in a list)
    if isinstance(video_input, list):
        # If it's a single video in a list, treat it as video
        if len(video_input) == 1 and video_input[0].name.endswith(
            (".mp4", ".avi", ".mov")
        ):
            video_input = video_input[0]  # unwrap single video file
        else:
            # Treat as list of images - copy and resize them
            frame_paths = []
            for idx, img_file in enumerate(video_input):
                img = Image.open(img_file.name).convert("RGB")
                # Resize if necessary
                width, height = img.size
                if max(width, height) > max_size:
                    scale = max_size / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                
                output_path = os.path.join(output_dir, f"frame_{idx:08d}.jpg")
                img.save(output_path, "JPEG", quality=95)
                frame_paths.append(output_path)
            return frame_paths

    # Handle file-like or path
    if hasattr(video_input, "name"):
        video_path = video_input.name
    elif isinstance(video_input, (str, os.PathLike)):
        video_path = str(video_input)
    else:
        raise ValueError(
            "Unsupported video input type. Must be a filepath, file-like object, or list of images."
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use ffmpeg to extract frames
    print(f"Extracting frames from video using ffmpeg...")
    try:
        # First, get video info to calculate frame interval
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', video_path
        ], capture_output=True, text=True, check=True)
        
        total_frames = int(result.stdout.strip())
        print(f"Total frames in video: {total_frames}")
        
        # Extract every k-th frame using ffmpeg with scaling
        ffmpeg_cmd = [
            'ffmpeg', '-i', video_path, '-y',
            '-vf', f'select=not(mod(n\\,{k})),scale=w=min({max_size}\\,iw):h=min({max_size}\\,ih):force_original_aspect_ratio=decrease',
            '-q:v', '2',  # High quality
            os.path.join(output_dir, 'frame_%08d.jpg')
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        # Get list of extracted frame paths
        frame_paths = sorted([
            os.path.join(output_dir, f) for f in os.listdir(output_dir) 
            if f.startswith('frame_') and f.endswith('.jpg')
        ])
        
        print(f"Extracted {len(frame_paths)} frames to {output_dir}")
        return frame_paths
        
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed: {e}")
        # Fallback to opencv if ffmpeg fails
        return extract_video_frames_fallback(video_path, output_dir, k, max_size)
    except FileNotFoundError:
        print("ffmpeg not found, using opencv fallback")
        return extract_video_frames_fallback(video_path, output_dir, k, max_size)


def extract_video_frames_fallback(video_path, output_dir, k=1, max_size=1024):
    """
    Fallback method using opencv, but saves frames to disk instead of memory.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video {video_path}.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    frame_paths = []

    os.makedirs(output_dir, exist_ok=True)

    with tqdm(total=total_frames // k, desc="Extracting Video Frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % k == 0:
                # Resize frame
                h, w = frame.shape[:2]
                scale = max(h, w) / max_size
                if scale > 1:
                    frame = cv2.resize(frame, (int(w / scale), int(h / scale)))
                
                # Save frame to disk
                frame_path = os.path.join(output_dir, f"frame_{len(frame_paths):08d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_paths.append(frame_path)
                pbar.update(1)
            frame_count += 1

    cap.release()
    return frame_paths


def resize_max_side(frame, max_size):
    """
    Resizes the frame so that its largest side equals max_size, maintaining aspect ratio.
    """
    height, width = frame.shape[:2]
    max_dim = max(height, width)

    if max_dim <= max_size:
        return frame  # No need to resize

    scale = max_size / max_dim
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_frame = cv2.resize(
        frame, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    return resized_frame


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def process_all_frames(
    IMG_FOLDER="/scratch/datasets/hq_data/night2_all_frames",
    to_visualize=False,
    save_images=True,
):
    dict_scores = {}
    for idx, img_name in tqdm(
        enumerate(sorted([x for x in os.listdir(IMG_FOLDER) if ".png" in x]))
    ):
        img = cv2.imread(os.path.join(IMG_FOLDER, img_name))  # [250:, 100:]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fm = (
            variance_of_laplacian(gray)
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.75, fy=0.75))
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.5, fy=0.5))
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.25, fy=0.25))
        )
        if to_visualize:
            plt.figure()
            plt.title(f"Laplacian score: {fm:.2f}")
            plt.imshow(img[..., [2, 1, 0]])
            plt.show()
        dict_scores[idx] = {"idx": idx, "img_name": img_name, "score": fm}
        if save_images:
            dict_scores[idx]["img"] = img

    return dict_scores


def select_optimal_frames(scores, k):
    """
    Selects a minimal subset of frames while ensuring no gaps exceed k.

    Args:
        scores (list of float): List of scores where index represents frame number.
        k (int): Maximum allowed gap between selected frames.

    Returns:
        list of int: Indices of selected frames.
    """
    n = len(scores)
    selected = [0, n - 1]
    i = 0  # Start at the first frame

    while i < n:
        # Find the best frame to select within the next k frames
        best_idx = max(
            range(i, min(i + k + 1, n)), key=lambda x: scores[x], default=None
        )

        if best_idx is None:
            break  # No more frames left

        selected.append(best_idx)
        i = best_idx + k + 1  # Move forward, ensuring gaps stay within k

    return sorted(selected)


def variance_of_laplacian(image):
    """
    Compute the variance of Laplacian as a focus measure.
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()


def preprocess_frame_paths(frame_paths, verbose=False):
    """
    Compute sharpness scores for a list of frame files using multi-scale Laplacian variance.

    Args:
        frame_paths (list of str): List of paths to frame image files.
        verbose (bool): If True, print scores.

    Returns:
        list of float: Sharpness scores for each frame.
    """
    scores = []

    for idx, frame_path in enumerate(tqdm(frame_paths, desc="Scoring frames")):
        # Load frame from disk
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not load frame {frame_path}")
            scores.append(0.0)
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fm = (
            variance_of_laplacian(gray)
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.75, fy=0.75))
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.5, fy=0.5))
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.25, fy=0.25))
        )

        if verbose:
            print(f"Frame {idx} ({os.path.basename(frame_path)}): Sharpness Score = {fm:.2f}")

        scores.append(fm)

    return scores


def preprocess_frames(frames, verbose=False):
    """
    Compute sharpness scores for a list of frames using multi-scale Laplacian variance.
    DEPRECATED: Use preprocess_frame_paths instead to avoid memory issues.

    Args:
        frames (list of np.ndarray): List of frames (BGR images).
        verbose (bool): If True, print scores.

    Returns:
        list of float: Sharpness scores for each frame.
    """
    scores = []

    for idx, frame in enumerate(tqdm(frames, desc="Scoring frames")):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fm = (
            variance_of_laplacian(gray)
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.75, fy=0.75))
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.5, fy=0.5))
            + variance_of_laplacian(cv2.resize(gray, (0, 0), fx=0.25, fy=0.25))
        )

        if verbose:
            print(f"Frame {idx}: Sharpness Score = {fm:.2f}")

        scores.append(fm)

    return scores


def select_optimal_frames(scores, k):
    """
    Selects k frames by splitting into k segments and picking the sharpest frame from each.

    Args:
        scores (list of float): List of sharpness scores.
        k (int): Number of frames to select.

    Returns:
        list of int: Indices of selected frames.
    """
    n = len(scores)
    selected_indices = []
    segment_size = n // k

    for i in range(k):
        start = i * segment_size
        end = (i + 1) * segment_size if i < k - 1 else n  # Last chunk may be larger
        segment_scores = scores[start:end]

        if len(segment_scores) == 0:
            continue  # Safety check if some segment is empty

        best_in_segment = start + np.argmax(segment_scores)
        selected_indices.append(best_in_segment)

    return sorted(selected_indices)


def copy_selected_frames_to_scene_dir(selected_frame_paths, scene_dir):
    """
    Copies selected frame files into the target scene directory under 'images/' subfolder.

    Args:
        selected_frame_paths (list of str): List of paths to selected frame files.
        scene_dir (str): Target path where 'images/' subfolder will be created.
    """
    import shutil
    
    images_dir = os.path.join(scene_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for idx, frame_path in enumerate(selected_frame_paths):
        filename = os.path.join(
            images_dir, f"{idx:08d}.jpg"
        )  # 00000000.jpg, 00000001.jpg, etc.
        shutil.copy2(frame_path, filename)

    print(f"Copied {len(selected_frame_paths)} selected frames to {images_dir}")


def save_frames_to_scene_dir(frames, scene_dir):
    """
    Saves a list of frames into the target scene directory under 'images/' subfolder.
    DEPRECATED: Use copy_selected_frames_to_scene_dir to avoid memory issues.

    Args:
        frames (list of np.ndarray): List of frames (BGR images) to save.
        scene_dir (str): Target path where 'images/' subfolder will be created.
    """
    images_dir = os.path.join(scene_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for idx, frame in enumerate(frames):
        filename = os.path.join(
            images_dir, f"{idx:08d}.png"
        )  # 00000000.png, 00000001.png, etc.
        cv2.imwrite(filename, frame)

    print(f"Saved {len(frames)} frames to {images_dir}")


def create_fallback_reconstruction(image_dir, sparse_path):
    """
    Create a minimal fallback reconstruction when COLMAP fails completely.
    Creates a simple linear camera trajectory for the available images.
    """
    # No need to import colmap_loader - we'll create text files directly
    
    print("🔧 Creating fallback reconstruction with assumed camera positions...")
    
    # Get list of images
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(image_files) < 1:
        raise RuntimeError("No images found for fallback reconstruction")
    
    # Read first image to get dimensions
    first_img_path = os.path.join(image_dir, image_files[0])
    from PIL import Image
    img = Image.open(first_img_path)
    width, height = img.size
    
    # Create minimal reconstruction directory
    fallback_dir = os.path.join(sparse_path, "0")
    os.makedirs(fallback_dir, exist_ok=True)
    
    # Create cameras.txt with simple pinhole model
    cameras_txt = os.path.join(fallback_dir, "cameras.txt")
    focal = max(width, height)  # Simple focal length estimation
    with open(cameras_txt, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {width} {height} {focal} {focal} {width/2} {height/2}\n")
    
    # Create images.txt with linear trajectory
    images_txt = os.path.join(fallback_dir, "images.txt")
    with open(images_txt, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i, img_file in enumerate(image_files):
            # Simple linear trajectory along Z-axis
            tx, ty, tz = 0.0, 0.0, -i * 0.5
            # Identity quaternion (no rotation)
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {img_file}\n")
            f.write("\n")  # Empty line for points2D
    
    # Create minimal points3D.txt with a few dummy points
    points_txt = os.path.join(fallback_dir, "points3D.txt")
    with open(points_txt, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        # Add some dummy 3D points for basic initialization
        for i in range(10):
            x, y, z = i * 0.1, 0.0, -1.0  # Simple grid of points
            r, g, b = 128, 128, 128  # Gray color
            error = 1.0
            f.write(f"{i+1} {x} {y} {z} {r} {g} {b} {error}\n")
    
    print(f"✅ Created fallback reconstruction with {len(image_files)} cameras at {fallback_dir}")
    print("⚠️  Note: This is a basic reconstruction with assumed camera positions. Results may be limited.")


def run_colmap_on_scene(scene_dir, force_pinhole=True, use_automatic_mode=False):
    """
    Runs feature extraction, matching, and mapping on all images inside scene_dir/images using pycolmap.
    Forces PINHOLE camera model to avoid distortion issues.

    Args:
        scene_dir (str): Path to scene directory containing 'images' folder.
        force_pinhole (bool): If True, forces PINHOLE camera model during reconstruction.
        use_automatic_mode (bool): If True, use settings similar to automatic_reconstructor.
    """
    start_time = time.time()
    print(f"Running COLMAP pipeline on all images inside {scene_dir}")
    if use_automatic_mode:
        print("Using automatic_reconstructor-like settings...")

    # Setup paths
    database_path = os.path.join(scene_dir, "database.db")
    sparse_path = os.path.join(scene_dir, "sparse")
    image_dir = os.path.join(scene_dir, "images")

    # Make sure output directories exist
    os.makedirs(sparse_path, exist_ok=True)

    # Step 1: Feature Extraction with automatic_reconstructor-like settings
    if use_automatic_mode:
        # Automatic reconstructor uses these defaults
        sift_options = {
            "max_num_features": 16384,  # Even higher for forest scenes
            "max_image_size": 3200,     # Much higher resolution
            "first_octave": -1,         # More detailed features
            "num_octaves": 4,
            "octave_resolution": 3,
            "peak_threshold": 0.001,    # Very sensitive feature detection
            "edge_threshold": 10,       # Standard edge threshold
            "estimate_affine_shape": False,
            "domain_size_pooling": False,
            "upright": False,
        }
    else:
        sift_options = {
            "max_num_features": 8192,
            "max_image_size": 1600,
            "first_octave": -1,
            "num_octaves": 4,
            "octave_resolution": 3,
            "peak_threshold": 0.005,
            "edge_threshold": 20,
        }
    
    pycolmap.extract_features(
        database_path,
        image_dir,
        sift_options=sift_options,
    )
    print(f"Finished feature extraction in {(time.time() - start_time):.2f}s.")

    # Step 2: Feature Matching
    if use_automatic_mode:
        # Try vocabulary tree matching first (similar to automatic_reconstructor)
        try:
            print("Attempting vocabulary tree matching...")
            # Note: This requires a vocabulary tree file which may not be available
            # Fall back to sequential matching if not available
            matching_options = pycolmap.SiftMatchingOptions()
            matching_options.max_ratio = 0.8
            matching_options.max_distance = 0.7
            matching_options.cross_check = True
            matching_options.max_num_matches = 32768
            
            # Use sequential matching for forest scenes (better for continuous motion)
            pycolmap.match_sequential(database_path, 
                                    sift_options=matching_options,
                                    overlap=30,  # Match with 30 neighboring frames for maximum overlap
                                    quadratic_overlap=True)  # Also match quadratically
            print("Sequential matching completed.")
            
            # Also do exhaustive matching for loop closure
            pycolmap.match_exhaustive(database_path, sift_options=matching_options)
            print("Exhaustive matching completed.")
        except Exception as e:
            print(f"Advanced matching failed, using exhaustive: {e}")
            sift_matching_options = pycolmap.SiftMatchingOptions()
            sift_matching_options.max_ratio = 0.8
            sift_matching_options.max_distance = 0.7
            sift_matching_options.cross_check = True
            pycolmap.match_exhaustive(database_path, sift_options=sift_matching_options)
    else:
        sift_matching_options = pycolmap.SiftMatchingOptions()
        sift_matching_options.max_ratio = 0.9
        sift_matching_options.max_distance = 0.8
        sift_matching_options.cross_check = True
        pycolmap.match_exhaustive(database_path, sift_options=sift_matching_options)
    
    print(f"Finished feature matching in {(time.time() - start_time):.2f}s.")

    # Step 3: Mapping with automatic_reconstructor-like parameters
    pipeline_options = pycolmap.IncrementalPipelineOptions()
    
    if use_automatic_mode:
        # Force single reconstruction for forest scenes
        pipeline_options.min_num_matches = 5           # Very low threshold to connect everything
        pipeline_options.multiple_models = False        # Disable multiple models entirely!
        pipeline_options.max_num_models = 1            # Force single model
        pipeline_options.max_model_overlap = 100       # Maximum overlap
        pipeline_options.min_model_size = 50           # Require large component
        pipeline_options.extract_colors = True
        pipeline_options.num_threads = -1              # Use all available threads
        
        # Mapper options optimized for forest/complex scenes
        pipeline_options.mapper.init_min_num_inliers = 15    # Lower threshold for connection
        pipeline_options.mapper.init_max_error = 8.0         # More lenient error tolerance
        pipeline_options.mapper.init_min_tri_angle = 2.0     # More lenient triangulation angle
        # pipeline_options.mapper.init_max_reg_trials = 3      # Not available in this pycolmap version
        
        pipeline_options.mapper.abs_pose_min_num_inliers = 15    # More lenient for connection
        pipeline_options.mapper.abs_pose_max_error = 12.0       # More lenient error
        pipeline_options.mapper.abs_pose_min_inlier_ratio = 0.15 # Lower ratio requirement
        # pipeline_options.mapper.abs_pose_min_num_correspondences = 3  # Not available in this pycolmap version
        
        pipeline_options.mapper.filter_max_reproj_error = 8.0    # More lenient filtering
        pipeline_options.mapper.filter_min_tri_angle = 0.25     # More lenient angle
        
        # pipeline_options.mapper.local_ba_num_images = 6  # May not be available
        # pipeline_options.mapper.local_ba_max_num_iterations = 25  # May not be available
        
        # Bundle adjustment refinement settings (may not all be available)
        try:
            pipeline_options.mapper.global_ba_refine_focal_length = True
            pipeline_options.mapper.global_ba_refine_principal_point = False
            pipeline_options.mapper.global_ba_refine_extra_params = False
        except AttributeError:
            pass  # Skip if not available
    else:
        # Original settings
        pipeline_options.min_num_matches = 8
        pipeline_options.multiple_models = True
        pipeline_options.max_num_models = 50
        pipeline_options.max_model_overlap = 20
        pipeline_options.min_model_size = 3
        pipeline_options.extract_colors = True
        pipeline_options.num_threads = 8
        
        pipeline_options.mapper.init_min_num_inliers = 15
        pipeline_options.mapper.init_max_error = 12.0
        pipeline_options.mapper.init_min_tri_angle = 2.0
        pipeline_options.mapper.abs_pose_min_num_inliers = 15
        pipeline_options.mapper.abs_pose_max_error = 12.0
        pipeline_options.mapper.filter_max_reproj_error = 8.0
        pipeline_options.mapper.filter_min_tri_angle = 1.5
    
    # Note: force_pinhole will be applied after reconstruction

    try:
        reconstruction = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=image_dir,
            output_path=sparse_path,
            options=pipeline_options,
        )
        print(f"Finished incremental mapping in {(time.time() - start_time):.2f}s.")
    except Exception as e:
        print(f"⚠️  Initial reconstruction failed: {e}")
        print("🔄 Trying with even more lenient settings...")
        
        # Try with ultra-lenient settings as fallback
        pipeline_options.min_num_matches = 5
        pipeline_options.min_model_size = 2
        pipeline_options.mapper.init_min_num_inliers = 10
        pipeline_options.mapper.init_max_error = 20.0
        pipeline_options.mapper.init_min_tri_angle = 1.0
        
        try:
            reconstruction = pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=image_dir,
                output_path=sparse_path,
                options=pipeline_options,
            )
            print(f"✅ Fallback reconstruction succeeded in {(time.time() - start_time):.2f}s.")
        except Exception as e2:
            print(f"❌ Both reconstruction attempts failed: {e2}")
            raise RuntimeError("COLMAP reconstruction failed. The video might have insufficient overlap or features.")

    # Step 4: Merge multiple reconstructions if they exist
    target_recon_path = os.path.join(sparse_path, "0")
    
    # Find all reconstructions
    reconstructions_found = []
    for i in range(100):  # Check more reconstruction indices
        alt_path = os.path.join(sparse_path, str(i))
        if os.path.exists(alt_path) and any(os.path.exists(os.path.join(alt_path, f)) 
                                         for f in ["cameras.bin", "images.bin", "points3D.bin"]):
            reconstructions_found.append((i, alt_path))
    
    if not reconstructions_found:
        print("❌ COLMAP reconstruction failed - creating minimal fallback reconstruction")
        return create_fallback_reconstruction(image_dir, sparse_path)
    
    print(f"🔍 Found {len(reconstructions_found)} reconstructions: {[i for i, _ in reconstructions_found]}")
    
    if len(reconstructions_found) == 1:
        # Single reconstruction - just ensure it's at index 0
        idx, recon_path = reconstructions_found[0]
        if idx != 0:
            import shutil
            if os.path.exists(target_recon_path):
                shutil.rmtree(target_recon_path)
            shutil.move(recon_path, target_recon_path)
            print(f"📁 Moved single reconstruction from {idx} to 0")
        recon_path = target_recon_path
    else:
        # Multiple reconstructions - try to merge them
        print("🔄 Attempting to merge multiple reconstructions...")
        try:
            merged_reconstruction = merge_colmap_reconstructions(reconstructions_found, sparse_path)
            merged_reconstruction.write(target_recon_path)
            print(f"✅ Successfully merged {len(reconstructions_found)} reconstructions")
            recon_path = target_recon_path
        except Exception as e:
            print(f"⚠️  Failed to merge reconstructions: {e}")
            # Fall back to using the largest reconstruction
            largest_idx, largest_path = max(reconstructions_found, 
                                          key=lambda x: get_reconstruction_size(x[1]))
            print(f"📊 Using largest reconstruction: {largest_idx}")
            if largest_idx != 0:
                import shutil
                if os.path.exists(target_recon_path):
                    shutil.rmtree(target_recon_path)
                shutil.move(largest_path, target_recon_path)
            recon_path = target_recon_path

    # Step 5: Convert cameras to PINHOLE if needed
    if os.path.exists(recon_path):
        reconstruction = pycolmap.Reconstruction(recon_path)
        
        if len(reconstruction.cameras) == 0:
            raise RuntimeError("❌ Reconstruction contains no cameras")
        if len(reconstruction.images) == 0:
            raise RuntimeError("❌ Reconstruction contains no images")
        if len(reconstruction.points3D) == 0:
            print("⚠️  Warning: Reconstruction contains no 3D points")

        for cam in reconstruction.cameras.values():
            if force_pinhole and cam.model != "PINHOLE":
                print(f"Converting camera {cam.camera_id} from {cam.model} to PINHOLE")
                cam.model = "PINHOLE"
                # Ensure we have exactly 4 parameters [fx, fy, cx, cy]
                if len(cam.params) >= 4:
                    cam.params = cam.params[:4]
                elif len(cam.params) >= 3:
                    # Duplicate focal length if we only have 3 params
                    f, cx, cy = cam.params[:3]
                    cam.params = [f, f, cx, cy]
                else:
                    # Default values if params are insufficient
                    focal = max(cam.width, cam.height)
                    cam.params = [focal, focal, cam.width/2, cam.height/2]

        reconstruction.write(recon_path)
        print(f"✅ Saved reconstruction with PINHOLE cameras to {recon_path}")
        print(f"📊 Reconstruction stats: {len(reconstruction.cameras)} cameras, {len(reconstruction.images)} images, {len(reconstruction.points3D)} points")

    print(f"Total pipeline time: {(time.time() - start_time):.2f}s.")


def process_input_for_colmap(input_path, num_ref_views, output_dir, max_size=1024, use_all_frames=False, target_fps=3.0):
    """
    Memory-efficient helper function to process video/images, select optimal frames,
    and save them to the output_dir/images without loading all frames into memory.
    
    Args:
        input_path: Path to video or image directory
        num_ref_views: Number of reference views to select (ignored if use_all_frames=True)
        output_dir: Output directory for processed images
        max_size: Maximum image dimension
        use_all_frames: If True, use all frames (or sample densely) instead of selecting optimal ones
    """
    import tempfile
    import shutil
    
    # Create temporary directory for extracted frames
    temp_frames_dir = tempfile.mkdtemp(prefix="edgs_frames_")
    
    try:
        if isinstance(input_path, (str, os.PathLike)):  # If input_path is a path string
            if os.path.isdir(input_path):  # If it's a directory of images
                print(f"Processing image directory: {input_path}")
                # Copy and resize images to temp directory
                frame_paths = []
                image_files = sorted([
                    f for f in os.listdir(input_path)
                    if f.lower().endswith(("jpg", "jpeg", "png"))
                ])
                
                for idx, img_file in enumerate(image_files):
                    img = Image.open(os.path.join(input_path, img_file)).convert("RGB")
                    # Resize if necessary
                    width, height = img.size
                    if max(width, height) > max_size:
                        scale = max_size / max(width, height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    output_path = os.path.join(temp_frames_dir, f"frame_{idx:08d}.jpg")
                    img.save(output_path, "JPEG", quality=95)
                    frame_paths.append(output_path)
                    
            else:  # If it's a single video file path
                print(f"Processing video file: {input_path}")
                frame_paths = extract_video_frames_to_disk(
                    video_input=input_path, 
                    output_dir=temp_frames_dir, 
                    max_size=max_size,
                    use_all_frames=use_all_frames
                )
        elif hasattr(input_path, "name"):  # File-like object (e.g., from Gradio upload)
            print(f"Processing uploaded video file: {input_path.name}")
            frame_paths = extract_video_frames_to_disk(
                video_input=input_path, 
                output_dir=temp_frames_dir, 
                max_size=max_size,
                use_all_frames=use_all_frames
            )
        else:
            raise ValueError(f"Unsupported input_path type: {type(input_path)}")

        if not frame_paths:
            print("No frames extracted or read.")
            return []

        if use_all_frames:
            # For automatic mode, extract frames at consistent rate to ensure good overlap
            total_frames = len(frame_paths)
            
            # Try to get video duration to calculate fps-based sampling
            try:
                # Get video info using ffprobe
                if isinstance(input_path, (str, os.PathLike)) and not os.path.isdir(input_path):
                    video_path = str(input_path)
                elif hasattr(input_path, "name"):
                    video_path = input_path.name
                else:
                    video_path = None
                    
                if video_path and os.path.exists(video_path):
                    import subprocess
                    result = subprocess.run([
                        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                        '-of', 'csv=p=0', video_path
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0 and result.stdout.strip():
                        duration = float(result.stdout.strip())
                        original_fps = total_frames / duration
                        
                        # Use the specified target_fps for frame extraction
                        actual_target_fps = min(original_fps, target_fps)  # Don't exceed original fps
                        target_frames = int(duration * actual_target_fps)
                        
                        if target_frames < total_frames:
                            # Calculate step size to maintain temporal consistency
                            step = total_frames / target_frames
                            selected_indices = [int(i * step) for i in range(target_frames)]
                            selected_frame_paths = [frame_paths[i] for i in selected_indices]
                            print(f"Extracted {len(selected_frame_paths)} frames at {actual_target_fps:.1f} fps from {duration:.1f}s video (original: {original_fps:.1f} fps)")
                        else:
                            selected_frame_paths = frame_paths
                            print(f"Using all {len(selected_frame_paths)} frames (short video or low fps)")
                    else:
                        raise Exception("Could not get duration")
                else:
                    raise Exception("No video file path available")
                    
            except Exception as e:
                print(f"Could not determine video duration ({e}), using frame-based sampling...")
                # Fallback: sample more densely to ensure overlap
                # Assume 30fps video, extract frames to approximate target_fps
                assumed_fps = 30.0
                step = max(1, int(assumed_fps / target_fps))
                
                # But limit total frames to avoid memory issues
                max_frames = 500  # Maximum frames to process
                if len(frame_paths[::step]) > max_frames:
                    step = max(1, total_frames // max_frames)
                    
                selected_frame_paths = frame_paths[::step]
                print(f"Sampled {len(selected_frame_paths)} frames (every {step} frames from {total_frames} total, targeting ~{target_fps} fps)")
                else:
                    selected_frame_paths = frame_paths
                    print(f"Using all {len(selected_frame_paths)} frames for reconstruction")
        else:
            # Original behavior: Score and select optimal frames
            print(f"Scoring {len(frame_paths)} frames...")
            frames_scores = preprocess_frame_paths(frame_paths)
            
            # Select optimal frames
            selected_frames_indices = select_optimal_frames(
                scores=frames_scores, k=min(num_ref_views, len(frame_paths))
            )
            
            # Get paths to selected frames
            selected_frame_paths = [frame_paths[idx] for idx in selected_frames_indices]
            
            print(f"Selected {len(selected_frame_paths)} optimal frames out of {len(frame_paths)}")

        # Copy selected frames to scene directory
        copy_selected_frames_to_scene_dir(selected_frame_paths, output_dir)
        
        # Return empty list since we're not loading frames into memory anymore
        # The actual frames are saved to disk in the scene directory
        return []
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_frames_dir):
            shutil.rmtree(temp_frames_dir)
            print(f"Cleaned up temporary frame directory: {temp_frames_dir}")


def process_input_for_colmap_legacy(input_path, num_ref_views, output_dir, max_size=1024):
    """
    DEPRECATED: Original memory-intensive version.
    Helper function to read frames from video or image folder, select optimal ones,
    and save them to the output_dir/images.
    This is based on process_input from gradio_demo.py.
    Renamed to avoid potential confusion if 'process_input' is too generic.
    """
    frames_to_save_in_scene_dir = []
    if isinstance(input_path, (str, os.PathLike)):  # If input_path is a path string
        if os.path.isdir(input_path):  # If it's a directory of images
            print(f"Processing image directory: {input_path}")
            raw_frames = []
            image_files = sorted(
                [
                    f
                    for f in os.listdir(input_path)
                    if f.lower().endswith(("jpg", "jpeg", "png"))
                ]
            )
            for img_file in image_files:
                img = Image.open(os.path.join(input_path, img_file)).convert("RGB")
                # Resize if necessary, similar to video frames
                width, height = img.size
                if max(width, height) > max_size:
                    scale = max_size / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                raw_frames.append(np.array(img))
        else:  # If it's a single video file path
            print(f"Processing video file: {input_path}")
            raw_frames = read_video_frames(video_input=input_path, max_size=max_size)
    elif hasattr(
        input_path, "name"
    ):  # If input_path is a file-like object (e.g., from Gradio upload)
        print(f"Processing uploaded video file: {input_path.name}")
        raw_frames = read_video_frames(video_input=input_path.name, max_size=max_size)
    else:
        raise ValueError(f"Unsupported input_path type: {type(input_path)}")

    if not raw_frames:
        print("No frames extracted or read.")
        return []

    frames_scores = preprocess_frames(
        raw_frames
    )  # Assuming preprocess_frames takes list of numpy arrays
    selected_frames_indices = select_optimal_frames(
        scores=frames_scores, k=min(num_ref_views, len(raw_frames))
    )
    frames_to_save_in_scene_dir = [
        raw_frames[frame_idx] for frame_idx in selected_frames_indices
    ]

    # The 'output_dir' here is the scene_dir where 'images' subfolder will be created
    save_frames_to_scene_dir(frames=frames_to_save_in_scene_dir, scene_dir=output_dir)
    return frames_to_save_in_scene_dir  # Returns the list of selected frame data (numpy arrays)


def orchestrate_video_to_colmap_scene(
    input_path,
    num_ref_views,
    max_size=1024,
    base_work_dir="../outputs/processed_scenes",
    use_automatic_mode=False,
    target_fps=3.0,
):
    """
    Orchestrates the full video/image folder preprocessing pipeline:
    1. Creates a temporary scene directory.
    2. Reads frames, selects optimal ones, saves them.
    3. Runs COLMAP on the scene.
    Args:
        input_path (str or file-like): Path string, a Gradio file object, or a list (e.g., from gr.Examples).
        num_ref_views (int): Number of reference views to select.
        max_size (int): Maximum size for width or height after resizing.
        base_work_dir (str): Base directory for temporary scene directories.
        use_automatic_mode (bool): If True, use automatic_reconstructor-like settings.
    Returns:
        the list of selected frame image data and the path to the COLMAP processed scene directory.
        This is based on preprocess_input from gradio_demo.py.
    """
    actual_input_path_str = None
    input_name_part = "temp_scene"  # Default

    if hasattr(input_path, "name") and isinstance(
        input_path.name, str
    ):  # Gradio file object
        actual_input_path_str = input_path.name
        input_name_part = os.path.splitext(os.path.basename(input_path.name))[0]
    elif isinstance(input_path, (str, os.PathLike)):  # Direct path string
        actual_input_path_str = str(input_path)
        input_name_part = os.path.splitext(os.path.basename(input_path))[0]
    elif (
        isinstance(input_path, list) and input_path
    ):  # Handle list: take the first item.
        # gr.Examples often wraps the path in another list, e.g., [['path/to/example.mp4']]
        # So, we might need to unwrap it.
        first_item_candidate = input_path[0]
        if (
            isinstance(first_item_candidate, list) and first_item_candidate
        ):  # Check for nested list
            first_item = first_item_candidate[0]
        else:
            first_item = first_item_candidate

        if hasattr(first_item, "name") and isinstance(
            first_item.name, str
        ):  # Gradio file object in list
            actual_input_path_str = first_item.name
            input_name_part = os.path.splitext(os.path.basename(first_item.name))[0]
        elif isinstance(first_item, (str, os.PathLike)):  # Path string in list
            actual_input_path_str = str(first_item)
            input_name_part = os.path.splitext(os.path.basename(first_item))[0]
        else:
            print(f"Warning: Unsupported item type in input list: {type(first_item)}")
            return [], None
    else:
        print(f"Error: Unsupported input_path type: {type(input_path)}")
        return [], None

    if not actual_input_path_str:
        print("Error: Could not determine a valid input file path.")
        return [], None

    print(f"Orchestrating COLMAP scene from: {actual_input_path_str}")

    # Using a structured output directory instead of pure tempfile.mkdtemp for easier inspection
    # scene_dir_parent = tempfile.mkdtemp() # Original approach

    # Ensure base_work_dir exists
    os.makedirs(base_work_dir, exist_ok=True)
    # Create scene directory without timestamp
    scene_dir = os.path.join(base_work_dir, input_name_part)
    
    # If directory already exists, add a counter instead of timestamp
    counter = 1
    original_scene_dir = scene_dir
    while os.path.exists(scene_dir):
        scene_dir = f"{original_scene_dir}_{counter}"
        counter += 1

    os.makedirs(scene_dir, exist_ok=True)
    print(f"Created scene directory for COLMAP: {scene_dir}")

    # Process video/images to extract and select optimal frames
    selected_frames_data = process_input_for_colmap(
        actual_input_path_str, num_ref_views, scene_dir, max_size, 
        use_all_frames=use_automatic_mode, target_fps=target_fps
    )
    
    # Check if images were saved to scene directory
    images_dir = os.path.join(scene_dir, "images")
    if not os.path.exists(images_dir) or not os.listdir(images_dir):
        print(f"Frame processing failed for {input_path}. No images found in {images_dir}. Aborting COLMAP.")
        return [], None

    # Run COLMAP with PINHOLE camera model enforced
    run_colmap_on_scene(scene_dir, force_pinhole=True, use_automatic_mode=use_automatic_mode)  # Force PINHOLE to avoid distortion

    print(f"COLMAP processing complete for {scene_dir}")
    return selected_frames_data, scene_dir


def get_reconstruction_size(recon_path):
    """Get the size (number of images) of a reconstruction."""
    try:
        reconstruction = pycolmap.Reconstruction(recon_path)
        return len(reconstruction.images)
    except:
        return 0


def merge_colmap_reconstructions(reconstructions_list, sparse_path):
    """
    Merge multiple COLMAP reconstructions into a single one.
    
    Args:
        reconstructions_list: List of (index, path) tuples
        sparse_path: Path to sparse directory
    
    Returns:
        Merged pycolmap.Reconstruction object
    """
    print(f"🔄 Merging {len(reconstructions_list)} reconstructions...")
    
    # Load all reconstructions
    reconstructions = []
    total_images = 0
    total_points = 0
    
    for idx, recon_path in reconstructions_list:
        try:
            recon = pycolmap.Reconstruction(recon_path)
            if len(recon.images) > 0:  # Only include non-empty reconstructions
                reconstructions.append((idx, recon))
                total_images += len(recon.images)
                total_points += len(recon.points3D)
                print(f"  Reconstruction {idx}: {len(recon.images)} images, {len(recon.points3D)} points")
        except Exception as e:
            print(f"  ⚠️  Could not load reconstruction {idx}: {e}")
    
    if not reconstructions:
        raise RuntimeError("No valid reconstructions to merge")
    
    if len(reconstructions) == 1:
        print(f"Only one valid reconstruction found, using it directly")
        return reconstructions[0][1]
    
    # Start with the largest reconstruction as base
    base_idx, base_recon = max(reconstructions, key=lambda x: len(x[1].images))
    print(f"📊 Using reconstruction {base_idx} as base ({len(base_recon.images)} images)")
    
    # Create a new merged reconstruction starting from the base
    merged = pycolmap.Reconstruction()
    
    # Copy cameras from base reconstruction
    for camera_id, camera in base_recon.cameras.items():
        merged.add_camera(camera)
    
    # Keep track of image names to avoid duplicates
    added_image_names = set()
    image_id_offset = 0
    point_id_offset = 0
    
    # Add all reconstructions
    for recon_idx, recon in reconstructions:
        print(f"  Adding reconstruction {recon_idx}...")
        
        # Add cameras (skip if already exists)
        for camera_id, camera in recon.cameras.items():
            if camera_id not in merged.cameras:
                merged.add_camera(camera)
        
        # Add images with ID offset to avoid conflicts
        for image_id, image in recon.images.items():
            if image.name not in added_image_names:
                new_image_id = image_id + image_id_offset
                # Create new image with offset ID
                new_image = pycolmap.Image(
                    id=new_image_id,
                    name=image.name,
                    camera_id=image.camera_id,
                    qvec=image.qvec,
                    tvec=image.tvec
                )
                merged.add_image(new_image)
                added_image_names.add(image.name)
            else:
                print(f"    Skipping duplicate image: {image.name}")
        
        # Add 3D points with ID offset
        for point_id, point in recon.points3D.items():
            new_point_id = point_id + point_id_offset
            merged.add_point3D(new_point_id, point.xyz, point.track, point.color)
        
        # Update offsets for next reconstruction
        if recon.images:
            image_id_offset += max(recon.images.keys()) + 1
        if recon.points3D:
            point_id_offset += max(recon.points3D.keys()) + 1
    
    print(f"✅ Merged reconstruction: {len(merged.images)} images, {len(merged.points3D)} points")
    
    return merged
