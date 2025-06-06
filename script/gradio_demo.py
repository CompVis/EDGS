import argparse
import contextlib
import io
import os
import shutil
import sys
import tempfile
import time

import gradio as gr
import hydra
import numpy as np
import torch
from hydra import compose, initialize

# Add the project root directory to sys.path
# so that modules from 'source' can be imported.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from source.trainer import EDGSTrainer
from source.utils_aux import set_seed
from source.utils_preprocess import (
    orchestrate_video_to_colmap_scene,  # Import the new/refactored function
    run_colmap_on_scene,
)
from source.visualization import (
    generate_circular_camera_path,
    generate_fully_smooth_cameras_with_tsp,
    put_text_on_image,
    save_numpy_frames_as_mp4,
)

# Init RoMA model:
sys.path.append("../submodules/RoMa")
from romatch import roma_indoor

roma_model = roma_indoor(device="cuda:0")
roma_model.upsample_preds = False
roma_model.symmetric = False

STATIC_FILE_SERVING_FOLDER = "./served_files"
MODEL_PATH = None
os.makedirs(STATIC_FILE_SERVING_FOLDER, exist_ok=True)

trainer = None


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def capture_logs(func, *args, **kwargs):
    log_capture_string = io.StringIO()
    tee = Tee(sys.__stdout__, log_capture_string)
    with contextlib.redirect_stdout(tee):
        result = func(*args, **kwargs)
    return result, log_capture_string.getvalue()


# Training Pipeline
def run_training_pipeline(
    scene_dir,
    num_ref_views=16,
    num_corrs_per_view=20000,
    num_steps=1_000,
    mode_toggle="Ours (EDGS)",
):
    with initialize(config_path="./configs", version_base="1.1"):
        cfg = compose(config_name="train")

    scene_name = os.path.basename(scene_dir)
    model_output_dir = f"./outputs/{scene_name}_trained"

    cfg.wandb.mode = "disabled"
    cfg.gs.dataset.model_path = model_output_dir
    cfg.gs.dataset.source_path = scene_dir
    cfg.gs.dataset.images = "images"

    cfg.gs.opt.TEST_CAM_IDX_TO_LOG = 12
    cfg.train.gs_epochs = 30000

    if mode_toggle == "Ours (EDGS)":
        cfg.gs.opt.opacity_reset_interval = 1_000_000
        cfg.train.reduce_opacity = True
        cfg.train.no_densify = True
        cfg.train.max_lr = True

        cfg.init_wC.use = True
        cfg.init_wC.matches_per_ref = num_corrs_per_view
        cfg.init_wC.nns_per_ref = 1
        cfg.init_wC.num_refs = num_ref_views
        cfg.init_wC.add_SfM_init = False
        cfg.init_wC.scaling_factor = 0.00077 * 2.0

    set_seed(cfg.seed)
    os.makedirs(cfg.gs.dataset.model_path, exist_ok=True)

    global trainer
    global MODEL_PATH
    generator3dgs = hydra.utils.instantiate(cfg.gs, do_train_test_split=False)
    trainer = EDGSTrainer(
        GS=generator3dgs,
        training_config=cfg.gs.opt,
        device=cfg.device,
        log_wandb=cfg.wandb.mode != "disabled",
    )

    # Disable evaluation and saving
    trainer.saving_iterations = []
    trainer.evaluate_iterations = []

    # Initialize
    trainer.timer.start()
    start_time = time.time()
    trainer.init_with_corr(cfg.init_wC, roma_model=roma_model)
    time_for_init = time.time() - start_time

    viewpoint_cams = trainer.GS.scene.getTrainCameras()
    path_cameras = generate_fully_smooth_cameras_with_tsp(
        existing_cameras=viewpoint_cams,
        n_selected=6,  # 8
        n_points_per_segment=30,  # 30
        closed=False,
    )
    path_cameras = path_cameras + path_cameras[::-1]

    path_renderings = []
    idx = 0
    # Visualize after init
    for _ in range(120):
        with torch.no_grad():
            viewpoint_cam = path_cameras[idx]
            idx = (idx + 1) % len(path_cameras)
            render_pkg = trainer.GS(viewpoint_cam)
            image = render_pkg["render"]
            image_np = np.clip(image.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            path_renderings.append(
                put_text_on_image(
                    img=image_np, text=f"Init stage.\nTime:{time_for_init:.3f}s.   "
                )
            )
    path_renderings = (
        path_renderings
        + [
            put_text_on_image(
                img=image_np, text=f"Start fitting.\nTime:{time_for_init:.3f}s.   "
            )
        ]
        * 30
    )

    # Train and save visualizations during training.
    start_time = time.time()
    for _ in range(int(num_steps // 10)):
        with torch.no_grad():
            viewpoint_cam = path_cameras[idx]
            idx = (idx + 1) % len(path_cameras)
            render_pkg = trainer.GS(viewpoint_cam)
            image = render_pkg["render"]
            image_np = np.clip(image.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            path_renderings.append(
                put_text_on_image(
                    img=image_np,
                    text=f"Fitting stage.\nTime:{time_for_init + time.time() - start_time:.3f}s.   ",
                )
            )

        cfg.train.gs_epochs = 10
        trainer.train(cfg.train)
        print(f"Time elapsed: {(time_for_init + time.time() - start_time):.2f}s.")
        # if (cfg.init_wC.use == False) and (time_for_init + time.time()-start_time) > 60:
        #     break
    final_time = time.time()

    # Add static frame. To highlight we're done
    path_renderings += [
        put_text_on_image(
            img=image_np,
            text=f"Done.\nTime:{time_for_init + final_time - start_time:.3f}s.   ",
        )
    ] * 30
    # Final rendering at the end.
    for _ in range(len(path_cameras)):
        with torch.no_grad():
            viewpoint_cam = path_cameras[idx]
            idx = (idx + 1) % len(path_cameras)
            render_pkg = trainer.GS(viewpoint_cam)
            image = render_pkg["render"]
            image_np = np.clip(image.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            path_renderings.append(
                put_text_on_image(
                    img=image_np,
                    text=f"Final result.\nTime:{time_for_init + final_time - start_time:.3f}s.   ",
                )
            )

    trainer.save_model()
    final_video_path = os.path.join(
        STATIC_FILE_SERVING_FOLDER, f"{scene_name}_final.mp4"
    )
    save_numpy_frames_as_mp4(
        frames=path_renderings, output_path=final_video_path, fps=30, center_crop=0.85
    )
    MODEL_PATH = cfg.gs.dataset.model_path
    ply_path = os.path.join(
        cfg.gs.dataset.model_path,
        f"point_cloud/iteration_{trainer.gs_step}/point_cloud.ply",
    )
    shutil.copy(
        ply_path, os.path.join(STATIC_FILE_SERVING_FOLDER, "point_cloud_final.ply")
    )

    return final_video_path, ply_path


# Gradio Interface
def gradio_interface(input_path, num_ref_views, num_corrs, num_steps):
    images, scene_dir = run_full_pipeline(
        input_path, num_ref_views, num_corrs, max_size=1024
    )
    shutil.copytree(
        scene_dir, STATIC_FILE_SERVING_FOLDER + "/scene_colmaped", dirs_exist_ok=True
    )
    (final_video_path, ply_path), log_output = capture_logs(
        run_training_pipeline, scene_dir, num_ref_views, num_corrs, num_steps
    )
    images_rgb = [img[:, :, ::-1] for img in images]
    return images_rgb, final_video_path, scene_dir, ply_path, log_output


# Dummy Render Functions
def render_all_views(scene_dir):
    viewpoint_cams = trainer.GS.scene.getTrainCameras()
    path_cameras = generate_fully_smooth_cameras_with_tsp(
        existing_cameras=viewpoint_cams,
        n_selected=8,
        n_points_per_segment=60,
        closed=False,
    )
    path_cameras = path_cameras + path_cameras[::-1]

    path_renderings = []
    with torch.no_grad():
        for viewpoint_cam in path_cameras:
            render_pkg = trainer.GS(viewpoint_cam)
            image = render_pkg["render"]
            image_np = np.clip(image.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            path_renderings.append(image_np)
    save_numpy_frames_as_mp4(
        frames=path_renderings,
        output_path=os.path.join(STATIC_FILE_SERVING_FOLDER, "render_all_views.mp4"),
        fps=30,
        center_crop=0.85,
    )

    return os.path.join(STATIC_FILE_SERVING_FOLDER, "render_all_views.mp4")


def render_circular_path(scene_dir):
    viewpoint_cams = trainer.GS.scene.getTrainCameras()
    path_cameras = generate_circular_camera_path(
        existing_cameras=viewpoint_cams, N=240, radius_scale=0.65, d=0
    )

    path_renderings = []
    with torch.no_grad():
        for viewpoint_cam in path_cameras:
            render_pkg = trainer.GS(viewpoint_cam)
            image = render_pkg["render"]
            image_np = np.clip(image.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            path_renderings.append(image_np)
    save_numpy_frames_as_mp4(
        frames=path_renderings,
        output_path=os.path.join(
            STATIC_FILE_SERVING_FOLDER, "render_circular_path.mp4"
        ),
        fps=30,
        center_crop=0.85,
    )

    return os.path.join(STATIC_FILE_SERVING_FOLDER, "render_circular_path.mp4")


# Download Functions
def download_cameras():
    path = os.path.join(MODEL_PATH, "cameras.json")
    return f"[📥 Download Cameras.json](file={path})"


def download_model():
    path = os.path.join(STATIC_FILE_SERVING_FOLDER, "point_cloud_final.ply")
    return f"[📥 Download Pretrained Model (.ply)](file={path})"


# Full pipeline helpers
def run_full_pipeline(input_path, num_ref_views, num_corrs, max_size=1024):
    tmpdirname = tempfile.mkdtemp()
    scene_dir = os.path.join(tmpdirname, "scene")
    os.makedirs(scene_dir, exist_ok=True)

    selected_frames = process_input(input_path, num_ref_views, scene_dir, max_size)
    run_colmap_on_scene(scene_dir)

    return selected_frames, scene_dir


def start_training(scene_dir, num_ref_views, num_corrs, num_steps):
    return capture_logs(
        run_training_pipeline, scene_dir, num_ref_views, num_corrs, num_steps
    )


# Gradio App
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=6):
            gr.Markdown(
                """
            ## <span style='font-size: 20px;'>📄 EDGS: Eliminating Densification for Efficient Convergence of 3DGS</span>
            🔗 <a href='https://compvis.github.io/EDGS' target='_blank'>Project Page</a>
            """,
                elem_id="header",
            )

    gr.Markdown(
        """
                ### <span style='font-size: 22px;'>🛠️ How to Use This Demo</span>

                1. Upload a **front-facing video** or **a folder of images** of a **static** scene.
                2. Use the sliders to configure the number of reference views, correspondences, and optimization steps.
                3. First press on preprocess Input to extract frames from video(for videos) and COLMAP frames.
                4. Then click **🚀 Start Reconstruction** to actually launch the reconstruction pipeline.
                5. Watch the training visualization and explore the 3D model.
                ‼️ **If you see nothing in the 3D model viewer**, try rotating or zooming — sometimes the initial camera orientation is off.


                ✅ Best for scenes with small camera motion.
                ❗ For full 360° or large-scale scenes, we recommend the Colab version (see project page).
                """,
        elem_id="quickstart",
    )

    scene_dir_state = gr.State()
    ply_model_state = gr.State()

    with gr.Row():
        with gr.Column(scale=2):
            input_file = gr.File(
                label="Upload Video or Images",
                file_types=[".mp4", ".avi", ".mov", ".png", ".jpg", ".jpeg"],
                file_count="multiple",
            )
            gr.Examples(
                examples=[
                    [["assets/examples/video_bakery.mp4"]],
                    [["assets/examples/video_flowers.mp4"]],
                    [["assets/examples/video_fruits.mp4"]],
                    [["assets/examples/video_plant.mp4"]],
                    [["assets/examples/video_salad.mp4"]],
                    [["assets/examples/video_tram.mp4"]],
                    [["assets/examples/video_tulips.mp4"]],
                ],
                inputs=[input_file],
                label="🎞️ ALternatively, try an Example Video",
                examples_per_page=4,
            )
            ref_slider = gr.Slider(
                4, 32, value=16, step=1, label="Number of Reference Views"
            )
            corr_slider = gr.Slider(
                5000,
                30000,
                value=20000,
                step=1000,
                label="Correspondences per Reference View",
            )
            fit_steps_slider = gr.Slider(
                100, 5000, value=400, step=100, label="Number of optimization steps"
            )
            preprocess_button = gr.Button("📸 Preprocess Input")
            start_button = gr.Button("🚀 Start Reconstruction", interactive=False)
            gallery = gr.Gallery(
                label="Selected Reference Views", columns=4, height=300
            )

        with gr.Column(scale=3):
            gr.Markdown("### 🏋️ Training Visualization")
            video_output = gr.Video(label="Training Video", autoplay=True)
            render_all_views_button = gr.Button("🎥 Render All-Views Path")
            render_circular_path_button = gr.Button("🎥 Render Circular Path")
            rendered_video_output = gr.Video(label="Rendered Video", autoplay=True)
        with gr.Column(scale=5):
            gr.Markdown("### 🌐 Final 3D Model")
            model3d_viewer = gr.Model3D(label="3D Model Viewer")

            gr.Markdown("### 📦 Output Files")
            with gr.Row(height=50):
                with gr.Column():
                    # gr.Markdown(value=f"[📥 Download .ply](file/point_cloud_final.ply)")
                    download_cameras_button = gr.Button("📥 Download Cameras.json")
                    download_cameras_file = gr.File(label="📄 Cameras.json")
                with gr.Column():
                    download_model_button = gr.Button(
                        "📥 Download Pretrained Model (.ply)"
                    )
                    download_model_file = gr.File(label="📄 Pretrained Model (.ply)")

    log_output_box = gr.Textbox(label="🖥️ Log", lines=10, interactive=False)

    def on_preprocess_click(input_file_obj, num_ref_views_val):
        """
        Handles the preprocess button click.
        Calls the main preprocessing orchestrator and updates the UI.
        """
        if input_file_obj is None:
            # Handles case where no file is uploaded if input_file component is not required
            # or if the user clears the selection.
            # For gr.Examples, input_file_obj will be a list containing a list of paths.
            # For direct upload with file_count="multiple", it's a list of file objects.
            # For direct upload with file_count="single", it's a single file object.
            # orchestrate_video_to_colmap_scene should be robust to these.
            gr.Warning("Please upload a file or select an example.")
            return None, None, gr.update(interactive=False)

        selected_bgr_frames, scene_dir = orchestrate_video_to_colmap_scene(
            input_path=input_file_obj,  # Pass the raw Gradio file object(s)
            num_ref_views=num_ref_views_val,
            max_size=1024,
            base_work_dir="./gradio_processed_scenes",  # Or configure as needed
        )

        if not scene_dir:  # Indicates preprocessing failed
            gr.Error("Preprocessing failed. Please check the logs or input file.")
            return (
                None,
                None,
                gr.update(interactive=False),
            )  # Keep gallery empty, scene_dir None, button disabled

        # Convert BGR numpy arrays to RGB for Gradio gallery.
        # gr.Gallery can display a list of NumPy arrays (H, W, C) or PIL Images.
        # Assuming selected_bgr_frames contains BGR NumPy arrays.
        gallery_display_images = []
        if selected_bgr_frames:
            gallery_display_images = [
                frame[..., ::-1]
                for frame in selected_bgr_frames
                if isinstance(frame, np.ndarray)
            ]

        return (
            gr.update(value=gallery_display_images),
            scene_dir,  # Update the scene_dir_state
            gr.update(interactive=True),  # Enable the 'Start Reconstruction' button
        )

    def on_start_click(scene_dir, num_ref_views, num_corrs, num_steps):
        (video_path, ply_path), logs = start_training(
            scene_dir, num_ref_views, num_corrs, num_steps
        )
        return video_path, ply_path, logs

    preprocess_button.click(
        fn=on_preprocess_click,
        inputs=[input_file, ref_slider],
        outputs=[gallery, scene_dir_state, start_button],
    )

    start_button.click(
        fn=on_start_click,
        inputs=[scene_dir_state, ref_slider, corr_slider, fit_steps_slider],
        outputs=[video_output, model3d_viewer, log_output_box],
    )

    render_all_views_button.click(
        fn=render_all_views, inputs=[scene_dir_state], outputs=[rendered_video_output]
    )
    render_circular_path_button.click(
        fn=render_circular_path,
        inputs=[scene_dir_state],
        outputs=[rendered_video_output],
    )

    download_cameras_button.click(
        fn=lambda: os.path.join(MODEL_PATH, "cameras.json"),
        inputs=[],
        outputs=[download_cameras_file],
    )
    download_model_button.click(
        fn=lambda: os.path.join(STATIC_FILE_SERVING_FOLDER, "point_cloud_final.ply"),
        inputs=[],
        outputs=[download_model_file],
    )

    gr.Markdown(
        """
    ---
    ### <span style='font-size: 20px;'>📖 Detailed Overview</span>

    If you uploaded a video, it will be automatically cut into a smaller number of frames (default: 16).

    The model pipeline:
    1. 🧠 Runs PyCOLMAP to estimate camera intrinsics & poses (~3–7 seconds for <16 images).
    2. 🔁 Computes 2D-2D correspondences between views. More correspondences generally improve quality.
    3. 🔧 Optimizes a 3D Gaussian Splatting model for several steps.

    ### 🎥 Training Visualization
    You will see a visualization of the entire training process in the "Training Video" pane.

    ### 🌀 Rendering & 3D Model
    - Render the scene from a circular path of novel views.
    - Or from camera views close to the original input.

    The 3D model is shown in the right viewer. You can explore it interactively:
    - On PC: WASD keys, arrow keys, and mouse clicks
    - On mobile: pan and pinch to zoom

    🕒 Note: the 3D viewer takes a few extra seconds (~5s) to display after training ends.

    ---
    Preloaded models coming soon. (TODO)
    """,
        elem_id="details",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Gradio demo for EDGS preprocessing and 3D viewing."
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to launch the Gradio app on."
    )
    parser.add_argument(
        "--no_share",
        action="store_true",
        help="Disable Gradio sharing and assume local access (default: share=True)",
    )
    args = parser.parse_args()

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=not args.no_share)
