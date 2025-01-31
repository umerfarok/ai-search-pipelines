#!/bin/bash

# Get current user's UID and GID
export UID=$(id -u)
export GID=$(id -g)

# Create base directories
mkdir -p {models,cache}/{models,model_cache,s3_cache,temp}
mkdir -p cache/model_cache/{transformers,huggingface}
mkdir -p cache/model_cache/huggingface/datasets

# Set permissions for all directories
for dir in models cache; do
  sudo chown -R ${UID}:${GID} $dir
  sudo chmod -R 755 $dir
done

# Create environment file
cat > .env << EOF
UID=${UID}
GID=${GID}
EOF
