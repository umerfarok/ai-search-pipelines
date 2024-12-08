# base_config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
    ZERO_SHOT_MODEL = 'facebook/bart-large-mnli'
    BATCH_SIZE = 32
    MAX_TOKENS = 512
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class SearchConfig:
    MAX_RESULTS = 5
    MINIMUM_SCORE = 0.3
    CACHE_SIZE = 10

# Product categories mapping
PRODUCT_CATEGORIES = {
    "appliances": {
        "keywords": ["refrigerator", "washer", "dryer", "dishwasher", "freezer"],
        "patterns": [
            r"(top|bottom|side by side) (\w+)",
            r"(energy efficient|star rated) (\w+)",
            r"(large capacity|counter depth) (\w+)"
        ]
    }
}