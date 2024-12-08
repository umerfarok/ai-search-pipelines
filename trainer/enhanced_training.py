# enhanced_training.py
from typing import List, Dict
import pandas as pd
import numpy as np
from transformers import pipeline

def generate_query_variations(product_data: pd.DataFrame) -> List[Dict]:
    """Generate various natural language queries for products"""
    
    # Initialize zero-shot classifier for categories
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    variations = []
    templates = [
        "I need something for {condition}",
        "What can I use for {condition}?",
        "I want to {purpose}, what should I use?",
        "Can you recommend something for {condition}?",
        "What's good for {purpose}?",
        "I'm looking for products that can help with {condition}",
        "What products would you recommend for {purpose}?",
        "I need help with {condition}, what should I use?",
        "What can I apply for {condition}?",
        "I want to {purpose}, need product recommendations"
    ]
    
    for _, row in product_data.iterrows():
        # Extract potential conditions and purposes from product description
        conditions = classifier(
            row['description'],
            candidate_labels=["pain", "headache", "muscle ache", "cooking", "cleaning"],
            multi_label=True
        )
        
        purposes = classifier(
            row['description'],
            candidate_labels=["pain relief", "cooking", "cleaning", "treatment", "care"],
            multi_label=True
        )
        
        # Generate variations
        for condition in conditions['labels'][:3]:
            for template in templates:
                if "{condition}" in template:
                    query = template.format(condition=condition)
                    variations.append({
                        "query": query,
                        "product_id": row['id'],
                        "relevance": 1.0,
                        "type": "condition"
                    })
                    
        for purpose in purposes['labels'][:3]:
            for template in templates:
                if "{purpose}" in template:
                    query = template.format(purpose=purpose)
                    variations.append({
                        "query": query,
                        "product_id": row['id'],
                        "relevance": 1.0,
                        "type": "purpose"
                    })
    
    return variations

def enhance_product_data(products: List[Dict]) -> List[Dict]:
    """Enhance product data with additional metadata"""
    enhanced_products = []
    
    for product in products:
        # Extract key information
        enhanced_product = product.copy()
        enhanced_product['metadata'] = enhanced_product.get('metadata', {})
        
        # Add usage scenarios
        enhanced_product['metadata']['usage_scenarios'] = extract_usage_scenarios(
            product['description']
        )
        
        # Add conditions it helps with
        enhanced_product['metadata']['conditions'] = extract_conditions(
            product['description']
        )
        
        # Add purposes/use cases
        enhanced_product['metadata']['purposes'] = extract_purposes(
            product['description']
        )
        
        enhanced_products.append(enhanced_product)
    
    return enhanced_products

def extract_usage_scenarios(text: str) -> List[str]:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(
        text,
        candidate_labels=[
            "pain relief", "cooking", "cleaning", "treatment",
            "personal care", "health", "food preparation"
        ],
        multi_label=True
    )
    return [label for label, score in zip(result['labels'], result['scores']) if score > 0.3]

def extract_conditions(text: str) -> List[str]:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(
        text,
        candidate_labels=[
            "headache", "muscle pain", "joint pain", "cooking needs",
            "cleaning needs", "skin condition", "general health"
        ],
        multi_label=True
    )
    return [label for label, score in zip(result['labels'], result['scores']) if score > 0.3]

def extract_purposes(text: str) -> List[str]:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(
        text,
        candidate_labels=[
            "pain relief", "cooking", "cleaning", "treatment",
            "care", "maintenance", "preparation"
        ],
        multi_label=True
    )
    return [label for label, score in zip(result['labels'], result['scores']) if score > 0.3]