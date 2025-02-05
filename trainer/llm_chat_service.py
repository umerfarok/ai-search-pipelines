import os
import logging
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import boto3
from botocore.exceptions import ClientError
import json
from pymongo import MongoClient
from bson import ObjectId
import redis
from datetime import datetime
from config import AppConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ModelManager:
    def __init__(self):
        self.mongo_client = MongoClient(AppConfig.MONGO_URI)
        self.db = self.mongo_client[AppConfig.MONGO_DB]
        self.versions_collection = self.db.model_versions
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=AppConfig.AWS_ACCESS_KEY,
            aws_secret_access_key=AppConfig.AWS_SECRET_KEY,
            endpoint_url=AppConfig.AWS_ENDPOINT_URL
        )
        self.loaded_models = {}
        self.model_cache = {}

    def get_model(self, version_id: str):
        """Get model by version ID"""
        if version_id in self.loaded_models:
            return self.loaded_models[version_id]

        # Get version info
        version_info = self.versions_collection.find_one({"_id": ObjectId(version_id)})
        if not version_info:
            raise ValueError(f"Model version {version_id} not found")

        # Download model if needed
        model_path = self._ensure_model_local(version_id)
        
        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            self.loaded_models[version_id] = {
                "model": model,
                "tokenizer": tokenizer,
                "config": version_info
            }
            
            return self.loaded_models[version_id]
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _ensure_model_local(self, version_id: str) -> str:
        """Ensure model files are available locally"""
        local_path = f"./models/{version_id}"
        if not os.path.exists(local_path):
            os.makedirs(local_path, exist_ok=True)
            
            # Download from S3
            s3_prefix = f"models/llm/{version_id}"
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=AppConfig.S3_BUCKET,
                    Prefix=s3_prefix
                )
                
                # Download each file
                for obj in response.get('Contents', []):
                    file_key = obj['Key']
                    local_file = os.path.join(local_path, os.path.relpath(file_key, s3_prefix))
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    self.s3_client.download_file(AppConfig.S3_BUCKET, file_key, local_file)
            except ClientError as e:
                logger.error(f"Error downloading model from S3: {e}")
                raise
            
            return local_path

    def clear_model(self, version_id: str):
        """Remove model from memory"""
        if version_id in self.loaded_models:
            del self.loaded_models[version_id]
            torch.cuda.empty_cache()


class ChatManager:
    def __init__(self):
        self.model_manager = ModelManager()
        self.mongo_client = MongoClient(AppConfig.MONGO_URI)
        self.db = self.mongo_client[AppConfig.MONGO_DB]
        self.chats_collection = self.db.chat_sessions
        self.redis_client = redis.Redis(
            host=AppConfig.REDIS_HOST,
            port=AppConfig.REDIS_PORT,
            decode_responses=True
        )

    def create_session(self, version_id: str, user_id: str) -> str:
        """Create a new chat session"""
        session = {
            "version_id": version_id,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "messages": [],
            "status": "active"
        }
        result = self.chats_collection.insert_one(session)
        return str(result.inserted_id)

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get chat session by ID"""
        return self.chats_collection.find_one({"_id": ObjectId(session_id)})

    def add_message(self, session_id: str, role: str, content: str):
        """Add message to chat session"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        }
        self.chats_collection.update_one(
            {"_id": ObjectId(session_id)},
            {"$push": {"messages": message}}
        )

    def generate_response(self, version_id: str, messages: List[Dict], max_length: int = 1000) -> str:
        """Generate response using the model"""
        try:
            # Get model and tokenizer
            model_info = self.model_manager.get_model(version_id)
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]

            # Format conversation history
            conversation = self._format_conversation(messages)
            
            # Tokenize input
            inputs = tokenizer(conversation, return_tensors="pt", truncation=True)
            inputs = inputs.to(model.device)

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._extract_response(response, conversation)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _format_conversation(self, messages: List[Dict]) -> str:
        """Format conversation history for the model"""
        formatted = ""
        for msg in messages:
            if msg["role"] == "user":
                formatted += f"\nUser: {msg['content']}"
            else:
                formatted += f"\nAssistant: {msg['content']}"
        formatted += "\nAssistant:"
        return formatted

    def _extract_response(self, generated_text: str, prompt: str) -> str:
        """Extract the model's response from generated text"""
        response = generated_text[len(prompt):].strip()
        # Clean up any incomplete sentences at the end
        if response and not any(response.endswith(p) for p in ".!?"):
            response = ". ".join(response.split(".")[:-1]) + "."
        return response

# Initialize components
chat_manager = ChatManager()

@app.route("/chat/session", methods=["POST"])
def create_chat_session():
    """Create a new chat session"""
    try:
        data = request.json
        if not data.get("version_id") or not data.get("user_id"):
            return jsonify({"error": "Missing required fields"}), 400

        session_id = chat_manager.create_session(
            data["version_id"],
            data["user_id"]
        )
        return jsonify({"session_id": session_id})

    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/chat/message", methods=["POST"])
def send_message():
    """Send a message and get response"""
    try:
        data = request.json
        if not all(k in data for k in ["session_id", "message"]):
            return jsonify({"error": "Missing required fields"}), 400

        # Get session
        session = chat_manager.get_session(data["session_id"])
        if not session:
            return jsonify({"error": "Session not found"}), 404

        # Add user message
        chat_manager.add_message(data["session_id"], "user", data["message"])

        # Generate response
        response = chat_manager.generate_response(
            session["version_id"],
            session["messages"] + [{"role": "user", "content": data["message"]}]
        )

        # Add assistant response
        chat_manager.add_message(data["session_id"], "assistant", response)

        return jsonify({
            "response": response,
            "session_id": data["session_id"]
        })

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/chat/session/<session_id>")
def get_chat_session(session_id):
    """Get chat session history"""
    try:
        session = chat_manager.get_session(session_id)
        if session:
            session["_id"] = str(session["_id"])
            return jsonify(session)
        return jsonify({"error": "Session not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

def initialize_service():
    """Initialize the chat service"""
    try:
        # Ensure model directory exists
        os.makedirs("./models", exist_ok=True)
        logger.info("Chat service initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        return False

if __name__ == "__main__":
    if initialize_service():
        app.run(host="0.0.0.0", port=AppConfig.SERVICE_PORT)
    else:
        exit(1)