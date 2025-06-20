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
    libgl1

# Set working directory
WORKDIR /app

# Copy environment.ymml into image
COPY environment.yml .

# Copy the files to the container
COPY paul.py .
COPY testVideo.mp4 .
COPY matrix.py .

# Install Miniconda for x86-64 (AMD64)
 RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     && bash Miniconda3-latest-Linux-x86_64.sh -b \
     && rm Miniconda3-latest-Linux-x86_64.sh

# Set the path to the conda binary
ENV PATH=/root/miniconda3/bin:$PATH

# Create a conda environment from .yml file
RUN conda env create -f environment.yml

# Update the path to new environment
ENV PATH="/root/miniconda3/envs/detections/bin:$PATH"


# Initialize conda and set up the environment
RUN conda init bash && \ 
    echo "conda activate detections" >> ~/.bashrc && \
    . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate detections

# Set the entrypoint
ENTRYPOINT ["/bin/bash"]


