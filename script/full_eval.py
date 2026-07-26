#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:
    name = "EDGS_"
    common_args = " --quiet --eval --test_iterations -1 "
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene 
        experiment = name + scene
        os.system(f"python train.py  verbose=True gs.dataset.source_path={source} gs.dataset.model_path={args.output_path}/mipnerf/{scene} wandb.name={experiment} init_wC.use=True train.gs_epochs=30000 init_wC.matches_per_ref=25_000 init_wC.nns_per_ref=3 gs.dataset.images=images_4 init_wC.num_refs=180 train.no_densify=True")
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        experiment = name + scene
        os.system(f"python train.py  verbose=True gs.dataset.source_path={source} gs.dataset.model_path={args.output_path}/mipnerf/{scene} wandb.name={experiment} init_wC.use=True train.gs_epochs=30000  init_wC.matches_per_ref=25_000 init_wC.nns_per_ref=3 gs.dataset.images=images_2 init_wC.num_refs=180 train.no_densify=True")
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene 
        experiment = name + scene +"_tandt"
        os.system(f"python train.py  verbose=True gs.dataset.source_path={source} gs.dataset.model_path={args.output_path}/mipnerf/{scene} wandb.name={experiment} init_wC.use=True train.gs_epochs=30000  init_wC.matches_per_ref=15_000 init_wC.nns_per_ref=3 init_wC.num_refs=180 train.no_densify=True")
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        experiment = name + scene + "_db"
        os.system(f"python train.py  verbose=True gs.dataset.source_path={source} gs.dataset.model_path={args.output_path}/mipnerf/{scene} wandb.name={experiment} init_wC.use=True train.gs_epochs=30000 init_wC.matches_per_ref=15_000 init_wC.nns_per_ref=3 init_wC.num_refs=180 train.no_densify=True")


if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene )
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)

    all_outputs = []
    for scene in mipnerf360_outdoor_scenes:
        all_outputs.append(args.output_path + "/mipnerf/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_outputs.append(args.output_path + "/mipnerf/" + scene)
    for scene in tanks_and_temples_scenes:
        all_outputs.append(args.output_path + "/tandt/" + scene)
    for scene in deep_blending_scenes:
        all_outputs.append(args.output_path + "/db/" + scene)


    common_args = " --quiet --eval --skip_train"
    for scene, source, output in zip(all_scenes, all_sources, all_outputs):
        os.system("python ./submodules/gaussian-splatting/render.py --iteration 7000 -s " + source + " -m " + output + common_args)
        os.system("python ./submodules/gaussian-splatting/render.py --iteration 30000 -s " + source + " -m " + output + common_args)

if not args.skip_metrics:
    all_outputs = []
    for scene in mipnerf360_outdoor_scenes:
        all_outputs.append(args.output_path + "/mipnerf/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_outputs.append(args.output_path + "/mipnerf/" + scene)
    for scene in tanks_and_temples_scenes:
        all_outputs.append(args.output_path + "/tandt/" + scene)
    for scene in deep_blending_scenes:
        all_outputs.append(args.output_path + "/db/" + scene)
        
    scenes_string = ""
    for scene, output in zip(all_scenes, all_outputs):
        scenes_string += "\"" + output + "\" "

    os.system("python metrics.py -m " + scenes_string)