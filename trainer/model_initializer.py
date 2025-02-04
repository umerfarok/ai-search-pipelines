import logging
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import spacy
from tqdm import tqdm
from config import AppConfig

logger = logging.getLogger(__name__)

class ModelInitializer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_models = {}
        self.nlp = None
        self.reranker = None
        self.intent_classifier = None

    def initialize_all(self):
        """Initialize all required models"""
        logger.info("Starting model initialization...")
        
        try:
            # Initialize embedding models
            self._init_embedding_models()
            
            # Initialize NLP pipeline
            self._init_nlp()
            
            # Initialize reranker
            self._init_reranker()
            
            # Initialize intent classifier
            self._init_intent_classifier()
            
            logger.info("Model initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            return False

    def _init_embedding_models(self):
        """Initialize all configured embedding models"""
        logger.info("Initializing embedding models...")
        
        # Always initialize default model first
        self._load_embedding_model(AppConfig.DEFAULT_MODEL)
        

    def _load_embedding_model(self, model_name: str):
        """Load a specific embedding model"""
        try:
            logger.info(f"Loading embedding model: {model_name}")
            model = SentenceTransformer(model_name)
            model.to(self.device)
            self.embedding_models[model_name] = model
            # Warm up the model with a sample input
            _ = model.encode("Warm up text", convert_to_tensor=True)
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {str(e)}")
            raise

    def _init_nlp(self):
        """Initialize spaCy NLP model"""
        logger.info("Initializing spaCy NLP model...")
        try:
            # Try loading the model first
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model isn't found, download it
            logger.info("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Warm up
        _ = self.nlp("Warm up text")
        logger.info("NLP pipeline initialized")

    def _init_reranker(self):
        """Initialize cross-encoder reranker"""
        logger.info("Initializing reranker...")
        from sentence_transformers import CrossEncoder
        
        try:
            self.reranker = CrossEncoder(AppConfig.RERANKER_MODEL)
            # Warm up
            _ = self.reranker.predict([("query", "document")])
            logger.info("Reranker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {str(e)}")
            raise

    def _init_intent_classifier(self):
        """Initialize zero-shot intent classifier"""
        logger.info("Initializing intent classifier...")
        try:
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            # Warm up
            _ = self.intent_classifier(
                "Warm up text",
                candidate_labels=["test"],
                multi_label=True
            )
            logger.info("Intent classifier initialized")
        except Exception as e:
            logger.error(f"Failed to initialize intent classifier: {str(e)}")
            raise

    def get_embedding_model(self, model_name: str = None) -> SentenceTransformer:
        """Get an initialized embedding model"""
        if not model_name:
            model_name = AppConfig.DEFAULT_MODEL
            
        if model_name not in self.embedding_models:
            self._load_embedding_model(model_name)
            
        return self.embedding_models[model_name]

    def get_nlp(self):
        """Get the initialized NLP pipeline"""
        if not self.nlp:
            self._init_nlp()
        return self.nlp

    def get_reranker(self):
        """Get the initialized reranker"""
        if not self.reranker:
            self._init_reranker()
        return self.reranker

    def get_intent_classifier(self):
        """Get the initialized intent classifier"""
        if not self.intent_classifier:
            self._init_intent_classifier()
        return self.intent_classifier