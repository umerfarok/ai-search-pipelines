#!/bin/bash

# Create required directories
mkdir -p models
mkdir -p cache/model_cache/transformers
mkdir -p cache/model_cache/huggingface
mkdir -p cache/temp
mkdir -p cache/s3_cache

# Set permissions (replace 1000 with your actual UID if different)
sudo chown -R 1000:1000 models
sudo chown -R 1000:1000 cache
sudo chmod -R 755 models
sudo chmod -R 755 cache
