#!/bin/bash
git clone git@github.com:CompVis/EDGS.git --recursive 
cd EDGS
git submodule update --init --recursive 

conda create -y -n edgs python=3.10 pip
conda activate edgs

# Optionally set path to CUDA
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y

pip install -e submodules/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -e submodules/gaussian-splatting/submodules/simple-knn

# For COLMAP and pycolmap
# Optionally install original colmap but probably pycolmap suffices
# conda install conda-forge/label/colmap_dev::colmap
pip install pycolmap


pip install wandb hydra-core tqdm torchmetrics lpips matplotlib rich plyfile imageio imageio-ffmpeg
conda install numpy=1.26.4 -y -c conda-forge --override-channels

pip install -e submodules/RoMa
conda install anaconda::jupyter --yes

# Stuff necessary for gradio and visualizations
pip install gradio 
pip install plotly scikit-learn moviepy==2.1.1 ffmpeg
pip install open3d 

