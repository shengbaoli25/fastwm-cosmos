# Match the base image, Torch wheel family and host driver; see MIGRATION_zh.md.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}
ARG TORCH_CUDA=cu128
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG INSTALL_HARDWARE=0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/fastwam/src:/opt/fastwam/deploy \
    TORCHINDUCTOR_CACHE_DIR=/tmp/fastwam-cache/inductor \
    TRITON_CACHE_DIR=/tmp/fastwam-cache/triton \
    CUDA_CACHE_PATH=/tmp/fastwam-cache/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv ca-certificates libglib2.0-0 libgl1 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
       --only-binary=:all: \
       --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
       torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}

WORKDIR /opt/fastwam
COPY deploy/requirements.txt /tmp/requirements-deploy.txt
COPY deploy/requirements-hardware.txt /tmp/requirements-hardware.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-deploy.txt \
    && if [ "$INSTALL_HARDWARE" = "1" ]; then \
       python3 -m pip install --no-cache-dir --only-binary=:all: -r /tmp/requirements-hardware.txt; \
       fi

COPY src/ /opt/fastwam/src/
COPY third_party/cosmos-predict2.5/ /opt/fastwam/third_party/cosmos-predict2.5/
COPY deploy/ /opt/fastwam/deploy/
COPY scripts/cosmos/export_noaction_deployment.py /opt/fastwam/scripts/cosmos/export_noaction_deployment.py

EXPOSE 8765
ENTRYPOINT ["python3", "/opt/fastwam/deploy/serve_realman.py", "--host", "0.0.0.0", "--port", "8765"]
