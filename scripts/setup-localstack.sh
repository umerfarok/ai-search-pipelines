#!/bin/bash

# Create required directories with proper permissions
mkdir -p ./cache/models
mkdir -p ./localstack/data
mkdir -p ./localstack/s3_data

# Set permissions
chmod -R 755 ./cache
chmod -R 755 ./localstack
chown -R 1000:1000 ./cache
chown -R 1000:1000 ./localstack

echo "LocalStack directories created and permissions set"
