# AI-Powered Product Search Engine

An advanced AI-based search solution that uses fine-tuned language models and transformer architectures to understand natural language queries and product descriptions. Built with modern AI/ML technologies, Go, Python, and React.
[![Integration Tests](https://github.com/umerfarok/ai-search-pipelines/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/umerfarok/ai-search-pipelines/actions/workflows/integration-tests.yml)

## 🤖 AI Features

- **Natural Language Understanding**: Process natural language queries like "I need something to deep fry fish" or "Looking for mouse control products"
- **LLM Fine-tuning**: Automatically fine-tunes GPT models on your product catalog for domain-specific responses
- **Contextual Understanding**: Creates rich product context for better semantic understanding
- **Smart Embeddings**: Uses all-mpnet-base-v2 for high-quality semantic embeddings
- **Hybrid Search**: Combines semantic search with fine-tuned LLM responses
- **Adaptive Learning**: Supports model retraining and version management

## 🌟 Key Features

- **Natural Language Product Search**: 
  - Ask questions in plain English about product needs
  - Get detailed, context-aware responses
  - Smart product recommendations based on use cases

- **Model Management**: 
  - Fine-tune LLMs on your product data
  - Monitor training progress in real-time
  - Version control for models

- **Custom Training**:
  - CSV data import with flexible schema mapping
  - Automatic use case extraction
  - Progressive model improvement

## 🏗️ Technical Architecture

The project leverages modern AI/ML architecture with three main services:

### AI/ML Components

- **Transformer Models**: 
  - GPT-2 base model for fine-tuning
  - MPNet embeddings for semantic search
  - Optimized for product domain

- **Training Pipeline**:
  - Automatic data preprocessing
  - Use case extraction
  - Query-response pair generation
  - Fine-tuning with custom product context

- **Model Storage**:
  - Cached model loading
  - Shared model volumes
  - S3-compatible storage

### Service Architecture

1. **Training Service**
   - Fine-tunes LLM models
   - Generates product embeddings
   - Saves models to cache directory
   - Real-time progress monitoring

2. **Search Service**
   - Loads fine-tuned models
   - Processes natural language queries
   - Combines semantic search with LLM responses
   - Handles context-aware filtering

3. **API Service**
   - Manages model configurations
   - Coordinates training jobs
   - Handles model versioning
   - Routes search requests

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU (recommended) with CUDA support
- 8GB+ RAM
- 20GB+ disk space

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd product-search
```

2. Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

Required environment variables:
```bash
# API Service
MONGO_URI=mongodb://root:example@localhost:27017
REDIS_HOST=localhost
REDIS_PORT=6379
AWS_ACCESS_KEY=your_access_key
AWS_SECRET_KEY=your_secret_key
S3_BUCKET=your_bucket
AWS_REGION=us-east-1

# Training Service
SERVICE_PORT=5001
MODEL_NAME=all-mpnet-base-v2
BATCH_SIZE=128

# Search Service
SERVICE_PORT=5000
MODEL_CACHE_SIZE=2
MIN_SCORE=0.2

# New Variables
MODEL_CACHE_DIR=/app/model_cache
SHARED_MODELS_DIR=/app/shared_models
CUDA_VISIBLE_DEVICES=0  # For GPU support
```

### Running with Docker

Start all services using Docker Compose:
```bash
docker-compose up -d
```

### Manual Setup

1. Start API Service:
```bash
cd api
go mod download
go run main.go
```

2. Start Training Service:
```bash
cd trainer
pip install -r requirements.txt
python app.py
```

3. Start Search Service:
```bash
cd search
pip install -r requirements.txt
python app.py
```

4. Start Frontend:
```bash
npm install
npm run dev
```

## 📝 API Endpoints

### Configuration Management

- `POST /config` - Create new model configuration
- `GET /config/:id` - Get configuration details
- `GET /config` - List all configurations
- `GET /config/status/:id` - Get training status

### Training Management

- `GET /training/status/:id` - Get training progress
- `GET /queue` - View queued training jobs

### Search

- `POST /search` - Perform semantic search
  ```json
  {
    "query": "search term",
    "config_id": "config_id",
    "max_items": 10,
    "filters": {
      "category": "electronics",
      "price_range": [0, 1000]
    }
  }
  ```

## 🎯 Usage

1. **Create Model Configuration**
   - Upload product data CSV
   - Configure schema mapping
   - Set training parameters

2. **Monitor Training**
   - Track training progress
   - View model status
   - Check for any errors

3. **Perform Search**
   - Select trained model
   - Enter search query
   - Apply filters
   - View ranked results

## 📁 Project Structure

```
.
├── api/               # Go API service
├── trainer/           # Python training service
├── search/           # Python search service
├── models/           # Trained model storage
├── data/             # Data storage
├── localstack/       # LocalStack configuration
├── cache/            # Cache storage
└── docker-compose.yaml
```

## 💡 Configuration Options


## 📊 Monitoring

- Check training progress: `/training/status/:id`
- Monitor queue: `/queue`
- View model performance: Model info panel in UI

## 🔒 Security Considerations

- Implement proper authentication
- Secure API endpoints
- Validate file uploads
- Sanitize search inputs
- Apply rate limiting

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## 📧 Support

For support, email [umerfarooq.dev@gmail.com] or create an issue in the repository.


docker-compose logs -f  search-service
docker-compose logs -f training-service
docker-compose exec api sh
