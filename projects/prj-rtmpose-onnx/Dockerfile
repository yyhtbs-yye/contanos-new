FROM contanos:base-onnx-gpu

# Install PyYAML for configuration loading
SHELL ["/bin/bash", "-lc"]
RUN conda init bash

RUN conda activate onnx && pip install --no-cache-dir --root-user-action=ignore \
        gdown

WORKDIR /app

RUN conda activate onnx && pip install --root-user-action=ignore git+https://github.com/yyhtbs-yye/rtmlib_copy.git

# Copy application files
COPY rtmpose_main_yaml.py .
COPY rtmpose_worker.py .

RUN mkdir -p /root/.cache/rtmlib/hub/checkpoints/
RUN conda activate onnx && gdown 1a8lgcSW3yW0ZyGeaewRpwcPqm4SaENkG -O /root/.cache/rtmlib/hub/checkpoints/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip

# Command to run the service with YAML config
# CMD ["bash", "-lc", "conda activate onnx && \
#                      python rtmpose_main_yaml.py \
#                          --config /app/basic_config.yaml \
#                          --daemon_mode true \
# "] 