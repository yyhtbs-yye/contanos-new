FROM contanos:base-opencv-cpu

SHELL ["/bin/bash", "-lc"]
RUN conda init bash

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN conda activate cv2 && pip install --no-cache-dir --root-user-action=ignore \
        filterpy \
        loguru \
        lap

COPY boxmot/ ./boxmot/

# Copy application files
COPY bytetrack_main_yaml.py .
COPY bytetrack_worker.py .
