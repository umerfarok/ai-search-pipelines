"""
Simple LLM manager for generating product recommendations
with better reliability and quality control.
"""

import os
import logging
import threading
import torch
import contextlib
import re
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

class EnhancedLLMManager:
    """Enhanced LLM manager with better response generation and fallback options"""
     
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "gpt2"  # Default to GPT-2 for wide compatibility
        self.model = None
        self.tokenizer = None
        self.model_lock = threading.Lock()
        self.initialized = False
        self._initialize_model()
        
        # Response templates for fallback
        self.templates = {
            "water_filter": [
                "Based on your search for water filtration in the {context}, I recommend the {product}. This {category} product is designed to effectively clean and purify water, making it safe for drinking.",
                "For your need to clean water in a {context} environment, the {product} would work well. It's designed to remove contaminants and provide clean drinking water.",
                "If you need to clean water while in the {context}, the {product} is a good choice. It's effective at filtering out impurities and making water potable."
            ],
            "general": [
                "Based on your search, I recommend the {product}. This {category} product should meet your needs for {query_intent}.",
                "The {product} appears to be the best match for your requirements. This {category} item is well-suited for {query_intent}.",
                "For your needs, I'd suggest the {product}. It's a popular choice in the {category} category for {query_intent}."
            ]
        }

    def _initialize_model(self):
        """Initialize the language model with optimal settings"""
        try:
            logger.info(f"Loading language model: {self.model_name}")
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Set up generation configuration
            model_config = {
                "pad_token_id": self.tokenizer.eos_token_id,
                "max_length": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
            }
            
            # Load model with efficient configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32  # Use FP32 for compatibility
            ).to(self.device)
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.initialized = True
            logger.info(f"Successfully loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.initialized = False

    def is_initialized(self) -> bool:
        """Check if the model is initialized"""
        return self.initialized

    @contextlib.contextmanager
    def get_model(self):
        """Context manager to safely access the model"""
        with self.model_lock:
            yield self.model, self.tokenizer

    def generate_response(self, prompt: str, max_length: int = 150) -> str:
        """Generate text response with improved quality control"""
        if not self.initialized:
            return self.get_template_response("I need to filter water", None, "jungle")
        
        try:
            with self.get_model() as (model, tokenizer):
                # Encode prompt
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Generate with properly aligned parameters
                with torch.no_grad():
                    output_sequences = model.generate(
                        **inputs,
                        max_length=len(inputs["input_ids"][0]) + max_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode output
                generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
                
                # Remove the prompt from the output
                if prompt in generated_text:
                    response = generated_text[len(prompt):].strip()
                else:
                    response = generated_text.strip()
                
                # Clean up response
                response = self._clean_response(response)
                    
                return response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I found some products that might help filter water in outdoor environments. Please check the product list for more details."

    def _clean_response(self, text: str) -> str:
        """Clean and filter generated text"""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove references to forms and data fields
        text = re.sub(r'Required data fields.*', '', text)
        text = re.sub(r'Required field must be.*', '', text)
        
        # Remove any weird formatting or code
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\*.*?\*', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        
        # Stop at natural ending points if found
        end_markers = ["\n\n", "\n---", "Thank you", "See more", "Learn more"]
        for marker in end_markers:
            if marker in text:
                text = text.split(marker)[0]
        
        return text.strip()
        
    def get_template_response(self, query: str, product: Dict, context: str = None) -> str:
        """Generate a template-based response when model generation fails"""
        import random
        
        # Determine the template to use
        template_key = "water_filter" if "water" in query.lower() else "general"
        templates = self.templates[template_key]
        
        # Fill in template
        template = random.choice(templates)
        
        if product:
            product_name = product.get('name', '').split('|')[0].strip()
            category = product.get('category', 'recommended')
        else:
            product_name = "recommended water filter"
            category = "water filtration"
            
        query_intent = "cleaning water" if "water" in query.lower() else query.lower()
        context_value = context or "outdoor"
        
        # Format the template
        response = template.format(
            product=product_name,
            category=category,
            query_intent=query_intent,
            context=context_value
        )
        
        return response
