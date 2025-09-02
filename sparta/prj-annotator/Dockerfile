FROM contanos:base-opencv-cpu

SHELL ["/bin/bash", "-lc"]
RUN conda init bash

# Install system dependencies for OpenCV and other requirements
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgstreamer1.0-0 \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        gstreamer1.0-tools && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN conda activate cv2 && pip install --no-cache-dir --root-user-action=ignore \
        asyncio

# Copy application files
COPY annotator_main_yaml.py .
COPY annotator_worker.py .

COPY pelpers /app/pelpers
# Command to run the service with YAML config
# CMD ["python", "annotation_main_yaml.py", "--config", "/app/pose_estimation_config.yaml", "--daemon_mode"] 