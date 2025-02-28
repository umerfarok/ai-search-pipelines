"""Domain knowledge module for product search enhancement."""

# Product categories mapping to related terms and synonyms
PRODUCT_CATEGORIES = {
    "water filter": ["water purifier", "water filtration", "water cleaning", "portable filter", 
                    "survival filter", "clean drinking water", "potable water", "water treatment"],
    
    "kitchen appliance": ["cooking", "mixer", "blender", "food processor", "toaster", "microwave",
                         "oven", "refrigerator", "cooker", "kitchenware"],
    
    "cleaning supplies": ["cleaner", "detergent", "cleaning solution", "vacuum", "mop", 
                          "broom", "sanitizer", "disinfectant", "washing"],
    
    "pest control": ["insect repellent", "bug killer", "mosquito repellent", "rodent control",
                     "pest trap", "pest deterrent", "ant killer", "roach killer"],
    
    "outdoor gear": ["camping", "hiking", "backpacking", "travel", "adventure", "survival",
                    "portable", "wilderness", "outdoor living", "trekking"],
    
    "electronics": ["gadget", "device", "electronic", "tech", "technology", "digital",
                   "smart device", "computer", "laptop", "phone", "tablet"],
}

# Product feature mapping for domain-specific understanding
PRODUCT_FEATURES = {
    "water filter": {
        "functions": ["purify", "clean", "filter", "remove contaminants", "potable", "drinkable"],
        "environments": ["jungle", "outdoors", "camping", "hiking", "travel", "emergency", "survival"],
        "properties": ["portable", "lightweight", "fast", "effective", "reliable", "durable", "compact"]
    },
    "kitchen appliance": {
        "functions": ["cook", "blend", "mix", "bake", "toast", "heat", "refrigerate"],
        "properties": ["efficient", "fast", "multi-functional", "electric", "automatic", "easy to use"]
    },
    "cleaning supplies": {
        "functions": ["clean", "sanitize", "disinfect", "wash", "remove dirt", "polish"],
        "properties": ["effective", "strong", "gentle", "concentrated", "eco-friendly"]
    },
    "pest control": {
        "functions": ["repel", "kill", "trap", "deter", "eliminate", "control"],
        "targets": ["insects", "bugs", "mosquitoes", "rodents", "ants", "roaches", "flies"]
    }
}

# Query intent mapping for better understanding user needs
QUERY_INTENTS = {
    "purchase": ["buy", "purchase", "order", "get", "shop for", "looking to buy", "want to get"],
    "information": ["how does", "what is", "tell me about", "information on", "details about", "learn about"],
    "comparison": ["compare", "versus", "vs", "difference between", "better than", "which is best"],
    "recommendation": ["recommend", "suggest", "best", "top", "good", "great", "excellent"]
}

# Domain-specific term expansion for better matching
DOMAIN_TERM_EXPANSION = {
    "clean water": ["water filter", "water purifier", "water filtration system", "water treatment"],
    "jungle": ["outdoor", "wilderness", "forest", "tropical", "survival", "camping", "hiking"],
    "purify water": ["water filter", "water purification", "clean water", "potable water"],
    "cook": ["kitchen appliance", "stove", "oven", "cooking utensil", "cookware"],
    "clean": ["cleaning supplies", "cleaner", "cleaning products", "sanitize", "disinfect"],
    "bugs": ["pest control", "insect repellent", "pest repellent", "bug killer"]
}

def get_expanded_terms(query_term):
    """Get expanded terms for a query term using domain knowledge"""
    # Check for direct matches
    if query_term in DOMAIN_TERM_EXPANSION:
        return DOMAIN_TERM_EXPANSION[query_term]
    
    # Check for partial matches
    for key, expansions in DOMAIN_TERM_EXPANSION.items():
        if query_term in key or key in query_term:
            return expansions
            
    # Return empty list if no matches found
    return []

def get_related_categories(term):
    """Get categories related to a term using domain knowledge"""
    related_categories = []
    
    for category, related_terms in PRODUCT_CATEGORIES.items():
        if term in related_terms or term in category:
            related_categories.append(category)
            
    return related_categories

def get_feature_terms(category):
    """Get feature terms for a category using domain knowledge"""
    if category in PRODUCT_FEATURES:
        feature_dict = PRODUCT_FEATURES[category]
        feature_terms = []
        
        for feature_type, terms in feature_dict.items():
            feature_terms.extend(terms)
            
        return feature_terms
    
    return []
