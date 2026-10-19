# CUDA 12.8 runtime is required by the CUDA 12.8 PyTorch wheels.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/fastwam/src:/opt/fastwam/deploy

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv ca-certificates libglib2.0-0 libgl1 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cu128 \
       torch==2.7.1 torchvision==0.22.1

WORKDIR /opt/fastwam
COPY deploy/requirements.txt /tmp/requirements-deploy.txt
# Install the compact inference runtime and RealMan/RealSense Python bindings.
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-deploy.txt

COPY src/ /opt/fastwam/src/
COPY third_party/cosmos-predict2.5/ /opt/fastwam/third_party/cosmos-predict2.5/
COPY deploy/ /opt/fastwam/deploy/

EXPOSE 8765
ENTRYPOINT ["python3", "/opt/fastwam/deploy/serve_realman.py", "--host", "0.0.0.0", "--port", "8765"]
