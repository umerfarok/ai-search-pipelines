# Embedding Models Documentation

## Overview
The project uses several state-of-the-art embedding models for semantic search and text understanding. Each model has been chosen for specific use cases and performance characteristics.

## Models

# Model Information

| Model Name                  | Parameters | Model Size (Approx) | Speed                   | Accuracy |
|-----------------------------|------------|---------------------|-------------------------|----------|
| all-MiniLM-L6-v2            | 22M        | ~82MB               | ⚡ Fastest               | ✅ Good  |
| all-MiniLM-L12-v2           | 33M        | ~120MB              | ⏳ Slower than L6       | ✅✅ Better |
| all-mpnet-base-v2           | 110M       | ~420MB              | 🚀 Slower than MiniLM   | 🔥 High  |
| paraphrase-mpnet-base-v2    | 110M       | ~420MB              | 🚀 Slower than MiniLM   | 🔥 High  |

### 1. All-MiniLM-L6 (Default)
- **Path**: `sentence-transformers/all-MiniLM-L6-v2`
- **Use Case**: General-purpose semantic search
- **Characteristics**:
  - Size: 80MB
  - Max Sequence Length: 256 tokens
  - Embedding Dimension: 384
- **Advantages**:
  - Excellent speed-to-performance ratio
  - Lower memory footprint
  - Good for production deployments
- **Best For**: 
  - Real-time search applications
  - General product search
  - High-throughput scenarios

### 2. BGE Small
- **Path**: `BAAI/bge-small-en-v1.5`
- **Use Case**: Fast, efficient semantic search
- **Characteristics**:
  - Size: ~120MB
  - Embedding Dimension: 384
  - Optimized for English text
- **Advantages**:
  - Better performance than MiniLM for specific domains
  - Fast inference speed
  - Good for multilingual content
- **Best For**:
  - Product category matching
  - Quick similarity searches
  - Resource-constrained environments

### 3. BGE Base
- **Path**: `BAAI/bge-base-en-v1.5`
- **Use Case**: Balanced semantic understanding
- **Characteristics**:
  - Size: ~440MB
  - Embedding Dimension: 768
  - Enhanced context understanding
- **Advantages**:
  - Better semantic understanding than smaller models
  - Good balance of speed and accuracy
  - Strong performance on longer texts
- **Best For**:
  - Detailed product descriptions
  - Complex query understanding
  - Balance of quality and performance

### 4. BGE Large
- **Path**: `BAAI/bge-large-en-v1.5`
- **Use Case**: High-accuracy semantic search
- **Characteristics**:
  - Size: ~1.3GB
  - Embedding Dimension: 1024
  - Superior context understanding
- **Advantages**:
  - Best semantic understanding
  - Highest accuracy
  - Better handling of nuanced queries
- **Best For**:
  - Premium search experiences
  - Complex product relationships
  - When accuracy is critical

## Model Selection Guide

### When to Use Each Model

1. **All-MiniLM-L6 (Default)**
   - Quick product searches
   - High-traffic applications
   - Limited computing resources
   - General-purpose semantic matching

2. **BGE Small**
   - Resource-constrained environments
   - Need for multilingual support
   - Fast category matching
   - Basic semantic search

3. **BGE Base**
   - Medium-complexity queries
   - Balanced performance needs
   - Good accuracy requirements
   - Mixed content types

4. **BGE Large**
   - Complex product relationships
   - Nuanced query understanding
   - Premium search experience
   - Accuracy over speed

