# This is the base image for the OS
FROM ubuntu:20.04

# Set the timezone and locale
ENV TZ=America/New_York \
    DEBIAN_FRONTEND=noninteractive

# Set the shell to bash
SHELL ["/bin/bash", "--login", "-c"]

# Install base system packages
RUN apt-get update && apt-get install -y \
    wget \
    libgl1 \
    nano \
    libglib2.0-0

# Set working directory
WORKDIR /app

# Create a directory for our products and volume
RUN mkdir -p /app/results

# Copy the files to the container
COPY program.py .
COPY testVideo.mp4 .
COPY requirements.txt .

# Set the path to the conda binary
ENV PATH=/root/miniconda3/bin:$PATH

# Install conda for ARM64
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
    && bash Miniconda3-latest-Linux-aarch64.sh -b \
    && rm Miniconda3-latest-Linux-aarch64.sh 

# Create a conda environment
RUN conda create -n detections python=3.10 -y

# Initialize conda and set up the environment
RUN conda init bash && \ 
    echo "conda activate detections" >> ~/.bashrc && \
    . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate detections && \
    #needed on ARM64 as you cannot pip install triton on ARM
    conda install -c conda-forge triton -y && \             
    pip install -r /app/requirements.txt

# Set the entrypoint
ENTRYPOINT ["/bin/bash"] 