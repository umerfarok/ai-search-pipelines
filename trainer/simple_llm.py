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
        
        # Replace templates with more flexible concept-based templates
        self.templates = {
            "general": [
                "Based on your need for {concept}, I recommend the {product}. This {category} product is designed to {function}, making it suitable for your requirements.",
                "For your {concept} needs, the {product} would be a good choice. It's a {category} product that offers {benefit}.",
                "The {product} would meet your needs for {concept}. This {category} product provides {benefit} which addresses your requirements."
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
        
    def get_template_response(self, query: str, product: Dict, concepts: List[str] = None) -> str:
        """Generate a template-based response using concepts rather than hardcoded categories"""
        import random
        
        # Get a general template
        template = random.choice(self.templates["general"])
        
        # Determine concept to use
        concept = "products"  # Default
        function = "meet your needs"
        benefit = "quality features"
        
        # Extract concepts from query if not provided
        if not concepts:
            concepts = []
            if "water" in query.lower() and any(term in query.lower() for term in ["clean", "purify", "filter", "drink"]):
                concepts.append("water purification")
            elif "portable" in query.lower() or "carry" in query.lower():
                concepts.append("portable solutions")
            elif any(term in query.lower() for term in ["jungle", "forest", "wilderness", "outdoor"]):
                concepts.append("outdoor equipment")
            elif any(term in query.lower() for term in ["kitchen", "cook", "food"]):
                concepts.append("kitchen tools")
        
        # Use the first concept
        if concepts:
            concept = concepts[0]
            
            # Map concept to function and benefit
            if concept == "water purification":
                function = "clean and purify water efficiently"
                benefit = "safe drinking water wherever you need it"
            elif concept == "outdoor equipment":
                function = "help you in outdoor environments"
                benefit = "reliability in wilderness conditions"
            elif concept == "portable solutions":
                function = "be easily carried and transported"
                benefit = "convenience while traveling"
            elif concept == "kitchen tools":
                function = "help prepare meals efficiently"
                benefit = "convenient food preparation"
        
        if product:
            product_name = product.get('name', '').split('|')[0].strip()
            category = product.get('category', 'recommended')
        else:
            product_name = "recommended product"
            category = "featured"
            
        # Format the template with the extracted information
        response = template.format(
            product=product_name,
            category=category,
            concept=concept,
            function=function,
            benefit=benefit
        )
        
        return response
