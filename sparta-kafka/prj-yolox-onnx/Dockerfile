FROM contanos:base-onnx-gpu

# Install PyYAML for configuration loading
SHELL ["/bin/bash", "-lc"]
RUN conda init bash

RUN conda activate onnx && pip install --no-cache-dir --root-user-action=ignore \
        gdown

WORKDIR /app

RUN conda activate onnx && pip install --root-user-action=ignore git+https://github.com/yyhtbs-yye/rtmlib_copy.git

# Copy application files
COPY yolox_main_yaml.py .
COPY yolox_worker.py .

RUN mkdir -p /root/.cache/rtmlib/hub/checkpoints/
RUN conda activate onnx && gdown 1lvvJKmEI6XtFsOip8UEHSfKkwgMv4jub -O /root/.cache/rtmlib/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.zip

# Command to run the service with YAML config
# CMD ["bash", "-lc", "conda activate onnx && \
#                      python yolox_main_yaml.py \
#                          --config /config/pose_estimation_config.yaml \
#                          --daemon_mode true \
# "] 