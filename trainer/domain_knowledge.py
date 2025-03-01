"""Domain knowledge module for product search enhancement."""

# Product categories mapping to related terms and synonyms
PRODUCT_CATEGORIES = {
    "water filter": ["water purifier", "water filtration", "water cleaning", "portable filter", 
                    "survival filter", "clean drinking water", "potable water", "water treatment",
                    "water purification", "water bottle filter", "lifestraw", "water tablet",
                    "water sterilization", "water disinfection"],
    
    "kitchen appliance": ["cooking", "mixer", "blender", "food processor", "toaster", "microwave",
                         "oven", "refrigerator", "cooker", "kitchenware", "culinary tools",
                         "juicer", "coffee maker", "rice cooker", "pressure cooker", "air fryer"],
    
    "cleaning supplies": ["cleaner", "detergent", "cleaning solution", "vacuum", "mop", 
                          "broom", "sanitizer", "disinfectant", "washing", "stain remover",
                          "floor cleaner", "surface cleaner", "bathroom cleaner", "kitchen cleaner"],
    
    "pest control": ["insect repellent", "bug killer", "mosquito repellent", "rodent control",
                     "pest trap", "pest deterrent", "ant killer", "roach killer", "fly trap",
                     "insect spray", "mosquito net", "bug zapper", "ultrasonic repeller"],
    
    "outdoor gear": ["camping", "hiking", "backpacking", "travel", "adventure", "survival",
                    "portable", "wilderness", "outdoor living", "trekking", "expedition",
                    "compass", "tent", "sleeping bag", "flashlight", "emergency kit"],
    
    "electronics": ["gadget", "device", "electronic", "tech", "technology", "digital",
                   "smart device", "computer", "laptop", "phone", "tablet", "camera",
                   "headphones", "speaker", "smartwatch", "power bank", "charger"],
                   
    "health and wellness": ["medicine", "vitamin", "supplement", "first aid", "bandage", 
                           "pain relief", "health monitor", "fitness tracker", "medical kit",
                           "thermometer", "massage", "therapy", "health device"],
                           
    "home appliance": ["fan", "air conditioner", "water heater", "washing machine", 
                      "vacuum cleaner", "air purifier", "humidifier", "dehumidifier",
                      "electric iron", "sewing machine", "hair dryer", "water dispenser"]
}

# Product feature mapping for domain-specific understanding
PRODUCT_FEATURES = {
    "water filter": {
        "functions": ["purify", "clean", "filter", "remove contaminants", "potable", "drinkable", 
                     "sterilize", "disinfect", "kill bacteria", "remove parasites", "eliminate viruses"],
        "environments": ["jungle", "outdoors", "camping", "hiking", "travel", "emergency", "survival",
                        "wilderness", "forest", "mountain", "river", "lake", "stream", "tropical"],
        "properties": ["portable", "lightweight", "fast", "effective", "reliable", "durable", "compact",
                      "easy-to-use", "long-lasting", "reusable", "chemical-free", "manual", "gravity-fed"]
    },
    "kitchen appliance": {
        "functions": ["cook", "blend", "mix", "bake", "toast", "heat", "refrigerate", "chop", "grind", 
                     "juice", "brew", "stir", "whip", "steam", "fry", "roast", "grill"],
        "properties": ["efficient", "fast", "multi-functional", "electric", "automatic", "easy to use",
                      "energy-saving", "compact", "large capacity", "programmable", "stainless steel",
                      "dishwasher-safe", "non-stick", "digital", "manual", "quiet"]
    },
    "cleaning supplies": {
        "functions": ["clean", "sanitize", "disinfect", "wash", "remove dirt", "polish", "deodorize",
                     "scrub", "wipe", "mop", "dust", "vacuum", "remove stains", "remove grease"],
        "properties": ["effective", "strong", "gentle", "concentrated", "eco-friendly", "non-toxic",
                      "biodegradable", "fragrance-free", "multi-purpose", "antibacterial", "fast-acting"]
    },
    "pest control": {
        "functions": ["repel", "kill", "trap", "deter", "eliminate", "control", "prevent", "catch",
                     "block", "protect", "guard", "eradicate", "destroy", "drive away"],
        "targets": ["insects", "bugs", "mosquitoes", "rodents", "ants", "roaches", "flies", "spiders",
                   "termites", "mice", "rats", "bed bugs", "fleas", "ticks", "wasps", "cockroaches"],
        "properties": ["safe", "effective", "non-toxic", "child-safe", "pet-friendly", "long-lasting",
                      "fast-acting", "indoor", "outdoor", "natural", "chemical-free", "odorless"]
    },
    "outdoor gear": {
        "functions": ["protect", "shelter", "carry", "navigate", "light", "cook", "signal", "warm", 
                     "hydrate", "communicate", "survive", "store", "repair", "defend"],
        "environments": ["forest", "mountain", "desert", "jungle", "snow", "river", "beach", "wilderness"],
        "properties": ["lightweight", "waterproof", "durable", "portable", "compact", "multi-purpose",
                      "weather-resistant", "foldable", "adjustable", "comfortable", "insulated"]
    },
    "health and wellness": {
        "functions": ["heal", "treat", "prevent", "monitor", "relieve", "improve", "maintain",
                     "strengthen", "support", "measure", "track", "reduce", "enhance"],
        "targets": ["pain", "stress", "blood pressure", "temperature", "injuries", "muscles",
                   "immune system", "digestion", "skin", "heart", "lungs", "joints", "sleep"],
        "properties": ["effective", "natural", "fast-acting", "long-lasting", "gentle",
                      "non-invasive", "portable", "accurate", "easy-to-use", "convenient"]
    }
}

# Query intent mapping for better understanding user needs
QUERY_INTENTS = {
    "purchase": ["buy", "purchase", "order", "get", "shop for", "looking to buy", "want to get"],
    "information": ["how does", "what is", "tell me about", "information on", "details about", "learn about"],
    "comparison": ["compare", "versus", "vs", "difference between", "better than", "which is best"],
    "recommendation": ["recommend", "suggest", "best", "top", "good", "great", "excellent"],
    "emergency": ["urgent", "emergency", "quickly", "immediately", "asap", "need right now"],
    "problem_solving": ["solve", "fix", "help with", "issue with", "problem with", "trouble with"],
    "situation_specific": ["for jungle", "while hiking", "during camping", "for travel", "in wilderness", 
                          "in emergency", "for survival", "in tropical", "during expedition"]
}

# Domain-specific term expansion for better matching
DOMAIN_TERM_EXPANSION = {
    "clean water": ["water filter", "water purifier", "water filtration system", "water treatment", 
                   "portable filter", "water purification tablet", "water sterilization", "potable water"],
    "jungle": ["outdoor", "wilderness", "forest", "tropical", "survival", "camping", "hiking", 
              "expedition", "adventure", "remote", "wild", "rainforest", "dense vegetation"],
    "purify water": ["water filter", "water purification", "clean water", "potable water", 
                    "remove contaminants", "kill bacteria", "remove parasites", "safe drinking"],
    "cook": ["kitchen appliance", "stove", "oven", "cooking utensil", "cookware", 
            "food preparation", "culinary", "baking", "grilling", "frying"],
    "clean": ["cleaning supplies", "cleaner", "cleaning products", "sanitize", "disinfect",
             "remove dirt", "wash", "wipe", "mop", "polish", "dust", "scrub"],
    "bugs": ["pest control", "insect repellent", "pest repellent", "bug killer", "insect spray",
            "mosquito repellent", "fly trap", "bug zapper", "insect deterrent"],
    "emergency": ["survival", "urgent", "immediate", "critical", "disaster", "first aid",
                 "emergency kit", "preparedness", "life-saving", "backup", "essential"],
    "portable": ["lightweight", "compact", "travel-sized", "mini", "small", "handheld",
                "foldable", "collapsible", "easy-to-carry", "space-saving", "travel-friendly"]
}

# Contextual situation mapping for better understanding of use cases
CONTEXTUAL_SITUATIONS = {
    "jungle": {
        "water_needs": ["portable filter", "water purifier", "sterilization tablets", "compact filter bottle"],
        "challenges": ["remote location", "contaminated water sources", "limited supplies", "emergency situation"],
        "priorities": ["effectiveness", "portability", "reliability", "ease of use", "no electricity required"]
    },
    "camping": {
        "water_needs": ["gravity filter", "pump filter", "filter bottle", "purification tablets"],
        "challenges": ["outdoor environment", "variable water sources", "limited carrying capacity"],
        "priorities": ["ease of use", "portability", "effectiveness", "durability", "group size capacity"]
    },
    "emergency": {
        "water_needs": ["quick solution", "ready-to-use", "reliable purification", "long shelf life"],
        "challenges": ["limited resources", "unknown water quality", "stress conditions", "no preparation"],
        "priorities": ["immediate results", "simplicity", "reliability", "effectiveness", "shelf stability"]
    },
    "travel": {
        "water_needs": ["compact solution", "reusable filter", "lightweight option", "easy to pack"],
        "challenges": ["different water quality standards", "limited space", "unknown sources"],
        "priorities": ["compact size", "ease of use", "effectiveness", "travel-friendly"]
    }
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

def get_situation_insights(context):
    """Get situation-specific insights for a context"""
    for situation, insights in CONTEXTUAL_SITUATIONS.items():
        if situation in context.lower() or context.lower() in situation:
            return insights
    return None
