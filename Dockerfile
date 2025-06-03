FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /EDGS

# Install system dependencies first, including git, build-essential, and cmake
RUN apt-get update && apt-get install -y \
  git \
  wget \
  build-essential \
  cmake \
  ninja-build \
  libgl1-mesa-glx \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Copy only essential files for cloning submodules first (e.g., .gitmodules)
# Or, if submodules are public, you might not need to copy anything specific for this step
# For simplicity, we'll copy everything, but this could be optimized
COPY . .

# Initialize and update submodules
RUN git submodule init && git submodule update --recursive

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
  bash /tmp/miniconda.sh -b -p /opt/conda && \
  rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Create the conda environment and install dependencies
RUN conda create -y -n edgs python=3.10 pip && \
  conda clean -afy && \
  echo "source activate edgs" > ~/.bashrc

# Set CUDA architectures to compile for
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"

# Activate the environment and install Python dependencies
RUN /bin/bash -c "source activate edgs && \
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
  pip install -e ./submodules/gaussian-splatting/submodules/diff-gaussian-rasterization && \
  pip install -e ./submodules/gaussian-splatting/submodules/simple-knn && \
  pip install pycolmap wandb hydra-core tqdm torchmetrics lpips matplotlib rich plyfile imageio imageio-ffmpeg && \
  pip install -e ./submodules/RoMa && \
  pip install gradio plotly scikit-learn moviepy==2.1.1 ffmpeg open3d"

# Expose the port for Gradio
EXPOSE 7862

# Command to run the Gradio demo
CMD ["bash", "-c", "source activate edgs && python gradio_demo.py --port 7862"]