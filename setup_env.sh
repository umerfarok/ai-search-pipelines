#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "=========================="
echo "Updating system packages..."
echo "=========================="
sudo apt update

########################################
# 1. Install Node.js 20+
########################################
echo ""
echo "========================================"
echo "Installing Node.js 20+ via NodeSource..."
echo "========================================"
# Download and run the Node.js 20.x setup script from NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
# Install Node.js (includes npm)
sudo apt install -y nodejs

# Verify Node.js and npm installation
echo "Node.js version: $(node -v)"
echo "npm version: $(npm -v)"

########################################
# 2. Install Docker Compose
########################################
# Check if docker-compose is already installed
if command -v docker-compose &>/dev/null; then
    echo ""
    echo "Docker Compose is already installed:"
    docker-compose --version
else
    echo ""
    echo "========================================"
    echo "Installing Docker Compose..."
    echo "========================================"
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose version: $(docker-compose --version)"
fi

########################################
# 3. Install NVIDIA Container Toolkit
########################################
echo ""
echo "========================================"
echo "Setting up NVIDIA Container Toolkit..."
echo "========================================"

# Add the NVIDIA package repositories
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
# Import the GPG key
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
# Add the repository to APT sources
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Update apt package list and install the NVIDIA container toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
echo ""
echo "Configuring Docker to use the NVIDIA runtime..."
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

# Restart Docker to apply the changes
sudo systemctl restart docker

echo ""
echo "Setup complete!"
