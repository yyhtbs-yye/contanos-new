FROM contanos:base-pytorch-gpu

# Install PyYAML for configuration loading
SHELL ["/bin/bash", "-lc"]
RUN conda init bash

RUN conda activate torch && pip install --no-cache-dir --root-user-action=ignore \
        gdown

WORKDIR /app

# Copy application files
COPY jerseyocr_main_yaml.py .
COPY jerseyocr_worker.py .
COPY pelpers/ ./pelpers/

# CMD ["bash", "-lc", "conda activate torch && python jerseyocr_main_yaml.py"]