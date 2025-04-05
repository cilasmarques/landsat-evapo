# Base image with CUDA support
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libtiff-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the source code
COPY . /app/

# Create output directory if it doesn't exist
RUN mkdir -p /app/output

# Set CUDA architecture - can be overridden during docker build
ARG CUDA_ARCH=sm_86

# Compile the application
RUN nvcc -arch=${CUDA_ARCH} -I ./include -g ./src/*.cu -o ./main -std=c++14 -ltiff -rdc=true

# Set the entrypoint
ENTRYPOINT ["/app/main"]

# Default command (can be overridden)
CMD ["--help"]