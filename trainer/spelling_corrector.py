from typing import Tuple, List
from collections import Counter
import re
from nltk.corpus import words
import nltk
from symspellpy import SymSpell, Verbosity

class SpellingCorrector:
    """Handles spelling correction using multiple approaches"""
    
    def __init__(self):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)
        self.word_set = set(words.words())
        self.custom_vocab = set()
        self.min_word_length = 3
        
    def initialize(self, custom_vocabulary: List[str] = None):
        """Initialize spelling corrector with custom vocabulary"""
        try:
            # Download required NLTK data if not already present
            nltk.download('words', quiet=True)
            
            # Add custom vocabulary
            if custom_vocabulary:
                self.custom_vocab.update(word.lower() for word in custom_vocabulary)
                
            # Build frequency dictionary for SymSpell
            for word in self.word_set | self.custom_vocab:
                self.sym_spell.create_dictionary_entry(word, 1)
                
        except Exception as e:
            logger.error(f"Failed to initialize spelling corrector: {e}")
            
    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize query into words"""
        return re.findall(r'\b\w+\b', query.lower())
    
    def _get_word_suggestions(self, word: str) -> List[Tuple[str, float]]:
        """Get spelling suggestions for a single word"""
        if len(word) < self.min_word_length:
            return [(word, 1.0)]
            
        if word in self.word_set or word in self.custom_vocab:
            return [(word, 1.0)]
            
        suggestions = self.sym_spell.lookup(
            word,
            Verbosity.ALL,
            max_edit_distance=2,
            include_unknown=True
        )
        
        return [(sugg.term, 1 - (sugg.distance / max(len(word), 1))) 
                for sugg in suggestions]
    
    def correct_spelling(self, query: str) -> Tuple[str, dict]:
        """
        Correct spelling in the query and return correction info
        
        Returns:
        Tuple[str, dict]: (corrected query, correction details)
        """
        tokens = self._tokenize_query(query)
        corrections = {}
        corrected_tokens = []
        
        for token in tokens:
            suggestions = self._get_word_suggestions(token)
            if suggestions:
                best_suggestion = max(suggestions, key=lambda x: x[1])
                if best_suggestion[0] != token:
                    corrections[token] = {
                        'corrected': best_suggestion[0],
                        'confidence': best_suggestion[1],
                        'alternatives': [s[0] for s in suggestions[1:3]]  # Top 2 alternatives
                    }
                corrected_tokens.append(best_suggestion[0])
            else:
                corrected_tokens.append(token)
        
        corrected_query = ' '.join(corrected_tokens)
        
        return corrected_query, {
            'original': query,
            'corrected': corrected_query,
            'corrections': corrections,
            'has_corrections': len(corrections) > 0
        }