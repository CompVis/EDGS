#!/usr/bin/env python
# coding: utf-8

# # EDGS: Eliminating Densification for Gaussian Splatting
# EDGS improves 3D Gaussian Splatting by removing the need for densification. It starts from a dense point cloud initialization based on 2D correspondences, leading to:
# - ⚡ Faster convergence (only 25% of training time)
#  - 🌀 Higher rendering quality
#  - 💡 No need for progressive densification

# ## 2. Import libraries
import argparse
import logging
import os
import random
import sys

import hydra
import numpy as np
import omegaconf
import torch
import wandb
from hydra import compose, initialize
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

# Add the project root directory to sys.path
# so that modules from 'source' can be imported.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# sys.path.append("../submodules/gaussian-splatting")
from source.trainer import EDGSTrainer
from source.utils_aux import set_seed
from source.utils_preprocess import (
    orchestrate_video_to_colmap_scene,  # Use the refactored function
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- Add argument parsing ---
parser = argparse.ArgumentParser(
    description="Fit EDGS model to a scene, optionally from a video."
)
parser.add_argument(
    "--video_path",
    type=str,
    default=os.path.join(
        project_root, "assets", "examples", "video_fruits.mp4"
    ),  # Use project_root
    help="Path to the input video file.",
)
parser.add_argument(
    "--num_ref_views",
    type=int,
    default=16,
    help="Number of reference views to extract from video for COLMAP.",
)
parser.add_argument(
    "--processed_scenes_dir",
    type=str,
    default=os.path.join(
        project_root, "outputs", "processed_scenes"
    ),  # Use project_root
    help="Base directory where processed COLMAP scenes will be stored.",
)
args = parser.parse_args()
# --- End argument parsing ---

with initialize(config_path="../configs", version_base="1.1"):
    cfg = compose(config_name="train")
print(OmegaConf.to_yaml(cfg))


# # 3. Init input parameters

# ## 3.1 Optionally preprocess video
# process the input video
if os.path.exists(args.video_path):
    print(f"Starting video processing for: {args.video_path}")
    try:
        # The first return value 'images_data' might not be directly used by the trainer
        # if the Scene object loads everything from the COLMAP directory.
        _, scene_dir = orchestrate_video_to_colmap_scene(
            args.video_path,
            args.num_ref_views,  # Assuming you added this arg
            max_size=1024,  # Or make it an arg
            base_work_dir=args.processed_scenes_dir,  # Assuming you added this arg
        )
        if scene_dir is None:
            print(f"Failed to process video {args.video_path}. Exiting.")
            sys.exit(1)
    except Exception as e:
        print(f"Error during video preprocessing: {e}")
        sys.exit(1)

# Update the config with your settings
cfg.gs.dataset.images = "images"
cfg.gs.opt.TEST_CAM_IDX_TO_LOG = 12
cfg.train.gs_epochs = 30000
cfg.gs.opt.opacity_reset_interval = 1_000_000
cfg.train.no_densify = True
cfg.init_wC.matches_per_ref = 15_000
cfg.init_wC.nns_per_ref = 3
cfg.init_wC.num_refs = 180
cfg.init_wC.roma_model = "outdoors"


# # 4. Initilize model and logger
if cfg.wandb.mode != "disabled":
    logging.info(
        "wandb logging is enabled (mode={}). Results will be logged to wandb.".format(
            cfg.wandb.mode
        )
    )
    _ = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
    )
else:
    logging.info(
        "wandb logging is disabled (mode={}). Results will not be logged to wandb.".format(
            cfg.wandb.mode
        )
    )
omegaconf.OmegaConf.resolve(cfg)
set_seed(cfg.seed)
# Init output folder
print("Output folder: {}".format(cfg.gs.dataset.model_path))
os.makedirs(cfg.gs.dataset.model_path, exist_ok=True)
# Init gs model
gs = hydra.utils.instantiate(cfg.gs)
trainer = EDGSTrainer(GS=gs, training_config=cfg.gs.opt, device=cfg.device)


# # 5. Init with matchings
trainer.timer.start()
trainer.init_with_corr(cfg.init_wC)
trainer.timer.pause()


# ### Visualize a few initial viewpoints
with torch.no_grad():
    viewpoint_stack = trainer.GS.scene.getTrainCameras()
    viewpoint_cams_to_viz = random.sample(trainer.GS.scene.getTrainCameras(), 4)
    for viewpoint_cam in viewpoint_cams_to_viz:
        render_pkg = trainer.GS(viewpoint_cam)
        image = render_pkg["render"]

        image_np = image.clone().detach().cpu().numpy().transpose(1, 2, 0)
        image_gt_np = (
            viewpoint_cam.original_image.clone()
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        # Clip values to be in the range [0, 1]
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        image_gt_np = np.clip(image_gt_np * 255, 0, 255).astype(np.uint8)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax[0].imshow(image_gt_np)
        ax[0].axis("off")
        ax[1].imshow(image_np)
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()


# # 6.Optimize scene
# Optimize first briefly for 5k steps and visualize results. We also disable saving of pretrained models. Train function can be changed for any other method
trainer.saving_iterations = []
cfg.train.gs_epochs = 5_000
trainer.train(cfg.train)


# ### Visualize same viewpoints
with torch.no_grad():
    for viewpoint_cam in viewpoint_cams_to_viz:
        render_pkg = trainer.GS(viewpoint_cam)
        image = render_pkg["render"]

        image_np = image.clone().detach().cpu().numpy().transpose(1, 2, 0)
        image_gt_np = (
            viewpoint_cam.original_image.clone()
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        # Clip values to be in the range [0, 1]
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        image_gt_np = np.clip(image_gt_np * 255, 0, 255).astype(np.uint8)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax[0].imshow(image_gt_np)
        ax[0].axis("off")
        ax[1].imshow(image_np)
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()


# ### Save model
with torch.no_grad():
    trainer.save_model()


# # 7. Continue training until we reach total 30K training steps
cfg.train.gs_epochs = 25_000
trainer.train(cfg.train)


# ### Visualize same viewpoints
with torch.no_grad():
    for viewpoint_cam in viewpoint_cams_to_viz:
        render_pkg = trainer.GS(viewpoint_cam)
        image = render_pkg["render"]

        image_np = image.clone().detach().cpu().numpy().transpose(1, 2, 0)
        image_gt_np = (
            viewpoint_cam.original_image.clone()
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
        )

        # Clip values to be in the range [0, 1]
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        image_gt_np = np.clip(image_gt_np * 255, 0, 255).astype(np.uint8)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax[0].imshow(image_gt_np)
        ax[0].axis("off")
        ax[1].imshow(image_np)
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()


# ### Save model
with torch.no_grad():
    trainer.save_model()
