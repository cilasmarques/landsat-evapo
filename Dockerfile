# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    make \
    git \
    wget \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add UbuntuGIS repository and install GDAL and TIFF dependencies
RUN add-apt-repository ppa:ubuntugis/ppa && \
    apt-get update && \
    apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libtiff5-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Make scripts executable
RUN chmod +x bin/*.sh scripts/*.sh

# Build the application
RUN make build

# Create necessary directories
RUN mkdir -p input output

# Set default environment variables
ENV METHOD=0
ENV OUTPUT_DATA_PATH=/app/output
ENV INPUT_DATA_PATH=/app/input

# Expose volumes for input and output
VOLUME ["/app/input", "/app/output"]

# Set the default command
CMD ["./main"] 