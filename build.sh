#!/bin/bash

# Set environment variables for local development
export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1

echo "Building ML base image..."
# Build and tag base image explicitly
docker-compose build ml-base
docker tag trainer_ml-base:latest ml-base:latest

echo "Building services..."
# Build services in parallel
docker-compose build --parallel search-service training-service api

echo "Build complete!"