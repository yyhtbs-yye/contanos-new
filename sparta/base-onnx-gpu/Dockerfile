# 1. Use NVIDIA CUDA 12.3 + cuDNN 9 runtime on Ubuntu 22.04
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

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

# 6. Create a dedicated conda environment called "onnx"
RUN conda create -y -n onnx python=3.10 && \
    conda clean -afy

# 7. Install Python dependencies via pip
RUN conda activate onnx && pip install --no-cache-dir \
        onnxruntime==1.22.0 \
        opencv-python==4.11.0.86 \
        paho-mqtt==2.1.0 \
        aiokafka==0.12.0 \
        av \
        tqdm \
        PyYAML

# Must install 'onnxruntime-gpu' seperatly!!!
RUN conda activate onnx && pip install onnxruntime-gpu==1.22.0

# Copy and install local contanos package
WORKDIR /opt/
RUN git clone https://github.com/yyhtbs-yye/contanos-core
WORKDIR /opt/contanos-core
RUN conda activate onnx && pip install --root-user-action=ignore .
