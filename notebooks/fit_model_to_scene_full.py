#!/usr/bin/env python
# coding: utf-8

# # EDGS: Eliminating Densification for Gaussian Splatting
# EDGS improves 3D Gaussian Splatting by removing the need for densification. It starts from a dense point cloud initialization based on 2D correspondences, leading to:
# - ⚡ Faster convergence (only 25% of training time)
#  - 🌀 Higher rendering quality
#  - 💡 No need for progressive densification

# ## 2. Import libraries
import argparse
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

sys.path.append("../")
sys.path.append("../submodules/gaussian-splatting")
from gradio_demo import preprocess_input
from source.trainer import EDGSTrainer
from source.utils_aux import set_seed

# --- Add argument parsing ---
parser = argparse.ArgumentParser(
    description="Fit EDGS model to a scene, optionally from a video."
)
parser.add_argument(
    "--video_path",
    type=str,
    default="../assets/examples/video_fruits.mp4",
    help="Path to the input video file.",
)
args = parser.parse_args()
# --- End argument parsing ---

with initialize(config_path="../configs", version_base="1.1"):
    cfg = compose(config_name="train")
print(OmegaConf.to_yaml(cfg))


# # 3. Init input parameters

# ## 3.1 Optionally preprocess video
PATH_TO_VIDEO = args.video_path
num_ref_views = 16  # how many frames you want to extract from video and colmap

# process the input video
if True:
    print("Starting video preprocessing...")
    # Ensure num_corrs is defined. Using cfg.init_wC.matches_per_ref as likely intended.
    num_corrs = cfg.init_wC.matches_per_ref
    try:
        images, scene_dir = preprocess_input(PATH_TO_VIDEO, num_ref_views, num_corrs)
        print(f"Video preprocessed. Scene directory: {scene_dir}")
        cfg.gs.dataset.source_path = scene_dir
        # Define a model_path, e.g., in a subdirectory of the scene_dir or a dedicated output folder
        cfg.gs.dataset.model_path = os.path.join(
            os.path.dirname(scene_dir), os.path.basename(scene_dir) + "_edgs_model"
        )
        print(f"Set dataset.source_path to: {cfg.gs.dataset.source_path}")
        print(f"Set dataset.model_path to: {cfg.gs.dataset.model_path}")
    except Exception as e:
        print(f"Error during video preprocessing: {e}")
        sys.exit(1)
else:
    # This block will be used if video preprocessing is skipped.
    # Ensure these paths are valid if this branch is taken.
    print("Skipping video preprocessing. Using pre-configured paths.")
    cfg.gs.dataset.model_path = "./scene_edgsed/"
    cfg.gs.dataset.source_path = (
        "../assets/scene_colmaped/"  # Ensure this is a valid COLMAP scene
    )

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
_ = wandb.init(
    entity=cfg.wandb.entity,
    project=cfg.wandb.project,
    config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    name=cfg.wandb.name,
    mode=cfg.wandb.mode,
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
