#!/bin/bash

set -e  # Exit on error

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Node.js (Latest LTS)
echo "Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs
node -v
npm -v

# Install Docker
echo "Installing Docker..."
sudo apt install -y ca-certificates curl gnupg
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Enable and start Docker service
sudo systemctl enable docker
sudo systemctl start docker

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Docker runtime..."
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify NVIDIA Docker installation
echo "Testing NVIDIA Docker..."
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Install Docker Compose (latest version)
echo "Installing Docker Compose..."
mkdir -p ~/.docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m) -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose

# Clone AI Search Pipelines Repository
echo "Cloning AI Search Pipelines repository..."
git clone https://github.com/umerfarok/ai-search-pipelines.git

# Verify installations
echo "Installed versions:"
node -v
npm -v
docker --version
docker compose version
nvidia-smi

echo "✅ All tools installed successfully!"
