#!/bin/bash

# Get the current user's UID and GID
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

# Create required directories
mkdir -p cache/models/transformers
mkdir -p cache/models/huggingface
mkdir -p cache/models/shared

# Set ownership and permissions
sudo chown -R $CURRENT_UID:$CURRENT_GID cache
chmod -R 755 cache

echo "Cache directories created and permissions set"
echo "UID:GID = $CURRENT_UID:$CURRENT_GID"

# Update the docker-compose.yml with current UID/GID
sed -i "s/user: \"1000:1000\"/user: \"$CURRENT_UID:$CURRENT_GID\"/" docker-compose.yml
