# AI Search & Training Architecture

## Core Services Overview

### Training Service
The training service handles LLM (Large Language Model) fine-tuning and model management.

#### Key Features
1. **Model Fine-tuning**
   - Base Model: Uses Mistral-7B as foundation
   - Training Data: Processes product descriptions and user queries
   - LoRA Adaptation: Efficient fine-tuning with low-rank adaptation

2. **Training Pipeline**
   ```
   Data Input → Preprocessing → Training → Model Storage
        ↓            ↓             ↓           ↓
     JSON/CSV    Tokenization   LoRA      S3 Storage
   ```

3. **Job Queue System**
   - Uses Redis for job queue management
   - Supports async training operations
   - Real-time progress tracking

### Search Service
The search service provides intelligent product search capabilities.

#### Components
1. **Query Understanding**
   - Natural language processing
   - Intent recognition
   - Spelling correction
   - Context extraction

2. **Hybrid Search**
   ```
   Query → Query Understanding → Hybrid Search → Ranking
    ↓           ↓                    ↓            ↓
   Text    Intent+Context     BM25+Semantic    Scores
   ```

3. **Features**
   - Semantic search
   - BM25 lexical search
   - Dynamic re-ranking
   - Context-aware results


### Training Flow


┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  API Service    │    │ Training Service│    │  Search Service │
│  (Go)          │◄─►│  (Python)      │◄─►│  (Python)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        ▲                      ▲                      ▲
        │                      │                      │
        ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    MongoDB      │    │     Redis       │    │  Model Cache    │
│  (Metadata)     │    │   (Queue)       │    │  (Local/S3)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘