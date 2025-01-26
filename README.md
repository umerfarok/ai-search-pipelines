# AI-Powered Product Search Engine

An advanced AI-based search solution that uses neural networks and transformer models to understand and match product descriptions semantically. Built with modern AI/ML technologies, Go, Python, and React.

## 🤖 AI Features

- **Neural Search**: Uses transformer-based deep learning models for semantic understanding
- **Contextual Understanding**: Understands product context beyond keyword matching
- **Smart Embeddings**: Generates vector embeddings using state-of-the-art language models
- **Intelligent Ranking**: AI-powered relevance scoring and result ranking
- **Adaptive Learning**: Support for model retraining and version management
- **Semantic Matching**: Understands similar products even with different descriptions

## 🌟 Features

- **Semantic Product Search**: Advanced search capabilities using sentence embeddings
- **Model Management**: Train and manage multiple search models
- **Real-time Training**: Queue-based training system with progress monitoring
- **Custom Schema Mapping**: Flexible product data mapping
- **Advanced Filtering**: Multi-dimensional filtering capabilities
- **Modern UI**: Responsive React-based interface

## 🏗️ Technical Architecture

The project leverages modern AI/ML architecture with three main services:

### AI/ML Components

- **Transformer Models**: Uses BERT/MPNet-based models for semantic understanding
- **Neural Embeddings**: Generates high-dimensional vector representations of products
- **Cosine Similarity**: Implements efficient vector similarity search
- **Batch Processing**: Optimized for large-scale data processing with GPU acceleration
- **Model Versioning**: Supports multiple AI models with different training configurations
- **Real-time Inference**: Fast search capabilities using optimized model loading

### Service Architecture

1. **API Service (Go)**
   - Handles configuration management
   - Manages model versions
   - Coordinates training jobs
   - Routes search requests

2. **Training Service (Python)**
   - Processes training jobs from queue
   - Generates embeddings using transformer models
   - Saves model artifacts to S3
   - Reports training progress

3. **Search Service (Python)**
   - Loads trained models
   - Performs semantic search
   - Handles filtering and ranking
   - Returns relevant results

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- Go 1.21+
- Python 3.9+
- Node.js 18+
- MongoDB
- Redis
- LocalStack (for local S3)

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
```env
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
- `GET /jobs/queue` - View queued training jobs

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

### Schema Mapping
```json
{
  "id_column": "product_id",
  "name_column": "title",
  "description_column": "description",
  "category_column": "category",
  "custom_columns": [
    {
      "user_column": "price",
      "standard_column": "price",
      "role": "filter"
    }
  ]
}
```

### Training Configuration
```json
{
  "model_type": "transformer",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "batch_size": 128,
  "max_tokens": 512,
  "validation_split": 0.2
}
```

## 🔍 Development

### Adding New Features

1. Create feature branch:
```bash
git checkout -b feature/new-feature
```

2. Make changes and test
3. Submit pull request

### Running Tests
```bash
# API Tests
cd api && go test ./...

# Training Service Tests
cd trainer && python -m pytest

# Search Service Tests
cd search && python -m pytest
```

## 📊 Monitoring

- Check training progress: `/training/status/:id`
- Monitor queue: `/jobs/queue`
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