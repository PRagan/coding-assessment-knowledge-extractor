"""
LLM Knowledge Extractor: Prototype that takes in text and uses an LLM to produce both a summary and structured data
Author: Philip Ragan
Service for manual text processing
"""

import re
import logging
from collections import Counter
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

logger = logging.getLogger(__name__)

class TextProcessor:
    """Service for manual text processing (non-LLM operations)"""
    
    def __init__(self):
        self._ensure_nltk_data()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # Fallback to basic stop words if NLTK data isn't available
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            logger.warning("Using basic stop words list - NLTK data not fully available")
    
    def _ensure_nltk_data(self):
        """Download required NLTK data if not present"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception as e:
                logger.warning(f"Could not download NLTK punkt: {e}")
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            try:
                nltk.download('stopwords', quiet=True)
            except Exception as e:
                logger.warning(f"Could not download NLTK stopwords: {e}")
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            try:
                nltk.download('averaged_perceptron_tagger', quiet=True)
            except Exception as e:
                logger.warning(f"Could not download NLTK POS tagger: {e}")
    
    def extract_keywords(self, text: str, top_k: int = 3) -> List[str]:
        """
        Extract the top_k most frequent nouns from text.
        Implements manual keyword extraction as required by the assignment.
        """
        
        # Validate text input
        if not text or not text.strip():
            return []
        
        try:
            # Method 1: Use NLTK for POS tagging (preferred)
            return self._extract_with_nltk(text, top_k)
        except Exception as e:
            logger.warning(f"NLTK extraction failed: {e}, falling back to regex method")
            # Method 2: Fallback to regex-based approach
            return self._extract_with_regex(text, top_k)
    
    def _extract_with_nltk(self, text: str, top_k: int) -> List[str]:
        """Extract nouns using NLTK POS tagging"""
        
        # Tokenize text
        try:
            tokens = word_tokenize(text.lower())
        except LookupError:
            # Fallback tokenization
            tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 2 and token.isalpha()
        ]
        
        # Get POS tags
        try:
            pos_tags = pos_tag(filtered_tokens)
            # Extract nouns (NN, NNS, NNP, NNPS)
            nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
        except Exception as e:
            logger.warning(f"POS tagging failed: {e}")
            # Fallback: use all filtered tokens
            nouns = filtered_tokens
        
        if not nouns:
            return []
        
        # Count frequency and get top_k
        noun_counts = Counter(nouns)
        top_nouns = [noun for noun, count in noun_counts.most_common(top_k)]
        
        return top_nouns
    
    # Extract nouns using regex patterns as a fallback
    def _extract_with_regex(self, text: str, top_k: int) -> List[str]:
        """Fallback method using regex patterns to identify likely nouns"""
        
        # Basic preprocessing
        text = text.lower()
        
        # Extract words (alphanumeric only)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        
        # Filter stop words
        filtered_words = [word for word in words if word not in self.stop_words]
        
        if not filtered_words:
            return []
        
        # Simple heuristic: words that appear multiple times are likely to be nouns
        word_counts = Counter(filtered_words)
        
        # Prioritize words that:
        # 1. Appear more than once
        # 2. Are longer than 4 characters
        # 3. Don't end with common verb suffixes
        
        scored_words = []
        for word, count in word_counts.items():
            score = count
            
            # Bonus for length
            if len(word) > 4:
                score += 1
            
            # Penalty for common verb endings
            if word.endswith(('ing', 'ed', 'er', 'ly')):
                score -= 1
            
            # Bonus for capitalization in original text (proper nouns)
            if word.title() in text or word.upper() in text:
                score += 1
            
            scored_words.append((word, score))
        
        # Sort by score and take top_k
        scored_words.sort(key=lambda x: x[1], reverse=True)
        top_words = [word for word, score in scored_words[:top_k]]
        
        return top_words
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for processing"""
        
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        return text.strip()
    
    # Get naive confidence scoring
    def calculate_confidence_score(self, text: str, summary: str, keywords: List[str]) -> float:
        """
        Calculate a naive confidence score for the analysis.
        This is a bonus feature with a simple heuristic.
        """
        
        score = 0.0
        
        # Base score for having content
        if text and summary:
            score += 0.3
        
        # Score based on text length (longer texts generally easier to analyze)
        text_length = len(text.split())
        if text_length > 50:
            score += 0.2
        elif text_length > 20:
            score += 0.1
        
        # Score based on keyword extraction success
        if keywords:
            score += 0.2 * (len(keywords) / 3.0)  # Up to 0.2 for 3 keywords
        
        # Score based on summary quality (simple heuristic)
        if summary and not summary.endswith('[Summary generated offline]'):
            score += 0.3
        else:
            score += 0.1  # Lower score for fallback summary
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, score))