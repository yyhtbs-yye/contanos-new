# 1. Use NVIDIA CUDA 12.1 + cuDNN 8 runtime on Ubuntu 22.04 (matches PyTorch cu121 wheels)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 2. Install system-level dependencies (git, build tools, FFmpeg, OpenCV prerequisites, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget \
      bzip2 \
      ca-certificates \
      git \
      build-essential \
      ffmpeg \
      libsm6 \
      libxext6 \
      libglib2.0-0 \
      pkg-config \
      libavformat-dev \
      libavcodec-dev \
      libavutil-dev && \
    rm -rf /var/lib/apt/lists/*

# 3. Install Miniconda3 into /opt/conda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    mkdir -p $CONDA_DIR && \
    bash /tmp/miniconda.sh -b -f -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy && \
    ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# 4. Initialize conda so that "conda activate" works in later layers
SHELL ["/bin/bash", "-lc"]
RUN conda init bash

RUN conda tos accept --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --channel https://repo.anaconda.com/pkgs/r

# 5. Create a dedicated conda environment called "torch"
RUN conda create -y -n torch python=3.11 && conda clean -afy

# 6. Install Python dependencies via pip
#    - PyTorch 2.2.1 / torchvision 0.17.1 / torchaudio 2.2.1 for CUDA 12.1
#    - OpenCV (Python)
RUN conda activate torch && \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 && \
    pip install --no-cache-dir \
      opencv-python==4.11.0.86 \
      paho-mqtt==2.1.0 \
      aiokafka==0.12.0 \
      av \
      tqdm \
      PyYAML

# 7. Copy and install local contanos package
WORKDIR /opt/
RUN git clone https://github.com/yyhtbs-yye/contanos-core
WORKDIR /opt/contanos-core
RUN conda activate torch && pip install --root-user-action=ignore .

