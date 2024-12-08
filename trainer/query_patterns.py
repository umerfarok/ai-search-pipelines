# query_patterns.py
from typing import Dict, List, Optional
import re

PRODUCT_CATEGORIES = {
    "appliances": {
        "keywords": [
            "refrigerator", "fridge", "freezer", "washer", "dryer", 
            "dishwasher", "microwave", "oven", "stove", "cooktop"
        ],
        "patterns": [
            r"(top|bottom|side by side) (\w+)",
            r"(large|small) capacity (\w+)",
            r"(energy efficient|energy star) (\w+)",
            r"(stainless steel|black|white) (\w+)",
            r"(\d+) cubic feet",
            r"counter depth (\w+)",
        ]
    },
    "kitchen": {
        "keywords": [
            "cook", "bake", "food", "kitchen", "cooking", "refrigeration",
            "storage", "cooling", "freezing", "preservation"
        ],
        "patterns": [
            r"(food|grocery) storage",
            r"(keep|store) food (\w+)",
            r"(cooking|baking|food prep) (\w+)",
        ]
    },
    "features": {
        "keywords": [
            "smart", "wifi", "connected", "energy star", "efficient",
            "ice maker", "water dispenser", "filter", "led", "digital"
        ],
        "patterns": [
            r"(smart|connected) (\w+)",
            r"(energy efficient|energy saving) (\w+)",
            r"with (ice maker|water dispenser)",
        ]
    },
    "brand": {
        "keywords": [
            "lg", "samsung", "whirlpool", "ge", "frigidaire", 
            "kitchenaid", "maytag", "bosch", "electrolux"
        ],
        "patterns": [
            r"(lg|samsung|whirlpool|ge|frigidaire) (\w+)",
            r"brand (\w+)",
        ]
    }
}

def get_category_keywords(category: str) -> List[str]:
    """Get keywords for a specific category."""
    return PRODUCT_CATEGORIES.get(category, {}).get("keywords", [])

def get_category_patterns(category: str) -> List[str]:
    """Get patterns for a specific category."""
    return PRODUCT_CATEGORIES.get(category, {}).get("patterns", [])

def extract_features_from_query(query: str) -> Dict[str, List[str]]:
    """Extract features and categories from a query."""
    features = {
        "categories": [],
        "keywords": [],
        "patterns": []
    }
    
    query = query.lower()
    
    for category, data in PRODUCT_CATEGORIES.items():
        # Check keywords
        for keyword in data["keywords"]:
            if keyword in query:
                features["categories"].append(category)
                features["keywords"].append(keyword)
                
        # Check patterns
        for pattern in data["patterns"]:
            matches = re.finditer(pattern, query)
            for match in matches:
                features["patterns"].extend(match.groups())
                
    return features

def enhance_query(query: str) -> str:
    """Enhance query with relevant keywords."""
    features = extract_features_from_query(query)
    additional_terms = []
    
    # Add category-specific keywords
    for category in features["categories"]:
        additional_terms.extend(get_category_keywords(category))
        
    # Add extracted pattern matches
    additional_terms.extend(features["patterns"])
    
    # Create enhanced query
    enhanced_terms = list(set(additional_terms))  # Remove duplicates
    if enhanced_terms:
        return f"{query} {' '.join(enhanced_terms)}"
    return query