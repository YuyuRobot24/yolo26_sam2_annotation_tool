FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/workspace/.config

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ultralytics
COPY src/ultralytics/ /opt/ultralytics/
RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

WORKDIR /opt/sam2
COPY src/sam2/ /opt/sam2/
# The runtime base does not include nvcc. SAM 2 supports inference without its
# optional connected-components CUDA extension.
RUN SAM2_BUILD_CUDA=0 python -m pip install -e .

WORKDIR /workspace
CMD ["bash"]
