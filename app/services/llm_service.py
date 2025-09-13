"""
LLM Knowledge Extractor: Prototype that takes in text and uses an LLM to produce both a summary and structured data
Author: Philip Ragan
Service for interacting with OpenAI's LLM API
"""

import os
import json
import logging
from typing import Dict, List, Optional
from openai import AsyncOpenAI
import asyncio

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with OpenAI's LLM API"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found. LLM functionality will be limited.")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Model configuration
        # self.model = "gpt-3.5-turbo"
        self.model = "gpt-5"
        self.max_retries = 3
        self.timeout = 30
    
    async def generate_summary(self, text: str) -> str:
        """Generate a 1-2 sentence summary of the input text"""
        
        if not self.client:
            return self._mock_summary(text)
        
        prompt = f"""
        Please provide a concise 1-2 sentence summary of the following text:
        
        Text: {text[:2000]}  # Limit text to avoid token limits
        
        Summary:
        """
        
        try:
            response = await self._make_api_call(prompt)
            summary = response.strip()
            
            # Ensure it's actually 1-2 sentences
            sentences = summary.split('.')
            if len(sentences) > 3:
                summary = '. '.join(sentences[:2]) + '.'
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return self._mock_summary(text)
    
    async def extract_metadata(self, text: str) -> Dict:
        """Extract structured metadata from text"""
        
        if not self.client:
            return self._mock_metadata(text)
        
        prompt = f"""
        Analyze the following text and extract metadata in JSON format.
        
        Text: {text[:2000]}
        
        Please extract:
        1. title: A suitable title for this text (if one exists or can be inferred)
        2. topics: Exactly 3 key topics or themes (as a list)
        3. sentiment: Overall sentiment (must be exactly one of: "positive", "neutral", "negative")
        
        Return only valid JSON in this format:
        {{
            "title": "extracted or inferred title or null",
            "topics": ["topic1", "topic2", "topic3"],
            "sentiment": "positive|neutral|negative"
        }}
        """
        
        try:
            response = await self._make_api_call(prompt)
            
            # Try to parse JSON response
            try:
                metadata = json.loads(response)
                
                # Validate and clean the response
                cleaned_metadata = self._validate_metadata(metadata)
                return cleaned_metadata
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON response from LLM: {response}")
                return self._mock_metadata(text)
                
        except Exception as e:
            logger.error(f"Failed to extract metadata: {str(e)}")
            return self._mock_metadata(text)
    
    async def _make_api_call(self, prompt: str) -> str:
        """Make API call to OpenAI with retries"""
        
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that provides concise, accurate responses."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.3
                    ),
                    timeout=self.timeout
                )
                
                return response.choices[0].message.content.strip()
                
            except asyncio.TimeoutError:
                logger.warning(f"API call timeout on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    raise Exception("API call timed out after all retries")
                    
            except Exception as e:
                logger.warning(f"API call failed on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise e
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
    
    def _validate_metadata(self, metadata: Dict) -> Dict:
        """Validate and clean metadata response"""
        
        cleaned = {}
        
        # Validate title
        title = metadata.get("title")
        if isinstance(title, str) and title.strip():
            cleaned["title"] = title.strip()
        else:
            cleaned["title"] = None
        
        # Validate topics (ensure exactly 3)
        topics = metadata.get("topics", [])
        if isinstance(topics, list):
            # Clean and limit to 3 topics
            cleaned_topics = [str(topic).strip() for topic in topics if str(topic).strip()][:3]
            # Pad with generic topics if needed
            while len(cleaned_topics) < 3:
                cleaned_topics.append(f"Topic {len(cleaned_topics) + 1}")
            cleaned["topics"] = cleaned_topics[:3]
        else:
            cleaned["topics"] = ["General", "Information", "Content"]
        
        # Validate sentiment
        sentiment = metadata.get("sentiment", "neutral").lower()
        if sentiment in ["positive", "negative", "neutral"]:
            cleaned["sentiment"] = sentiment
        else:
            cleaned["sentiment"] = "neutral"
        
        return cleaned
    
    def _mock_summary(self, text: str) -> str:
        """Generate a mock summary when LLM is unavailable"""
        words = text.split()
        if len(words) <= 20:
            return text
        return f"This text discusses {' '.join(words[:15])}... [Summary generated offline]"
    
    def _mock_metadata(self, text: str) -> Dict:
        """Generate mock metadata when LLM is unavailable"""
        words = text.lower().split()
        
        return {
            "title": None,
            "topics": ["General", "Content", "Information"],
            "sentiment": "neutral"
        }