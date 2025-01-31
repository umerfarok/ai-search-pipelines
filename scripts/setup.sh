#!/bin/bash

# Create cache directories
mkdir -p cache/models/transformers
mkdir -p cache/models/huggingface

# Set permissions
chmod -R 755 cache/models
chown -R 1000:1000 cache/models

echo "Cache directories created and permissions set"
