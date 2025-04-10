FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

LABEL maintainer="Hemant Sharma <hsharma@anl.gov>"
LABEL description="Docker image for LaueMatching"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenmpi-dev \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/AdvancedPhotonSource/LaueMatching .

# Create build directory
RUN mkdir -p build && cd build && \
    cmake .. -DBUILD_LIBRARY=ON && \
    make -j$(nproc) && \
    make download_orientation_file

# Install Python requirements
RUN pip3 install -r requirements.txt

# Set PATH to include the binaries
ENV PATH="/app/build/bin:${PATH}"
ENV LD_LIBRARY_PATH="/app/build/lib:${LD_LIBRARY_PATH}"

# Set up a volume for data
VOLUME /data

# Set working directory to /data
WORKDIR /data

# Default command
CMD ["bash"]