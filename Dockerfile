# Use a base Python image
FROM python:3.10-slim

# Install system dependencies and useful tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    bash-completion \
    vim \
    less \
    procps \
    htop \
    curl \
    git \
    wget \
    swig \
    libglu1-mesa-dev \
    xvfb \
    libgl1-mesa-dev \
    libosmesa6-dev \
    patchelf \
    ffmpeg \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Set up bash aliases
RUN echo "alias ll='ls -la'" >> /root/.bashrc

# Set the working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports for TensorBoard or other tools
EXPOSE 6006

# Default command
CMD ["bash"]
