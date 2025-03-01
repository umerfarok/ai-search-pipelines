#!/bin/bash

# Ensure cache directories exist and have correct permissions
mkdir -p /app/cache/transformers \
    /app/cache/huggingface \
    /app/cache/torch \
    /app/cache/nltk_data \
    /app/cache/onnx

# Set permissions
chmod -R 777 /app/cache

# Execute the main command
exec "$@"
