# Start from the NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set a fixed model cache directory.
ENV TORCH_HOME=/root/.cache/torch

# Install Python and necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential python3.10 python3-pip python3.10-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# update pip and setuptools
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# as described in the Unsloth.ai Github
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install torch==2.2.1 torchvision torchaudio 
RUN pip install "numpy==1.26.4"
RUN pip install --no-deps "xformers==0.0.25" "trl<0.9.0" peft accelerate bitsandbytes
RUN pip install triton

# Copy the requirements file into the container
COPY finetune.py .

CMD ["python3", "finetune.py"]
