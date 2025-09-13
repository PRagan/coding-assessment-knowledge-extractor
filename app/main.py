"""
LLM Knowledge Extractor: Prototype that takes in text and uses an LLM to produce both a summary and structured data
Author: Philip Ragan
Main FastAPI application with endpoints for text analysis and search
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import logging
from typing import List, Optional

try:
    # Try absolute imports (for Docker/production)
    from app.models import AnalysisRequest, AnalysisResponse, SearchResponse
    from app.services.llm_service import LLMService
    from app.services.text_processor import TextProcessor
    from app.database import get_db, init_db
    from app.crud import create_analysis, search_analyses
except ImportError:
    # Fall back to relative imports (for local development)
    from models import AnalysisRequest, AnalysisResponse, SearchResponse
    from services.llm_service import LLMService
    from services.text_processor import TextProcessor
    from database import get_db, init_db
    from crud import create_analysis, search_analyses

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Knowledge Extractor",
    description="Extract summaries and structured metadata from unstructured text using LLMs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db(clear_existing=True)
    logger.info("Database cleared and initialized successfully")

@app.get("/")
async def root():
    return {"message": "LLM Knowledge Extractor API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(
    request: AnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Analyze unstructured text and extract summary + structured metadata.
    """
    try:
        # Validate input
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text input provided")
        
        text = request.text.strip()
        logger.info(f"Analyzing text of length: {len(text)}")
        
        # Initialize services
        llm_service = LLMService()
        text_processor = TextProcessor()
        
        # Process text with error handling
        try:
            # Generate summary using LLM
            summary = await llm_service.generate_summary(text)
            
            # Extract structured metadata using LLM
            metadata = await llm_service.extract_metadata(text)
            
        except Exception as e:
            logger.error(f"LLM service error: {str(e)}")
            raise HTTPException(
                status_code=503, 
                detail="LLM service temporarily unavailable. Please try again later."
            )
        
        # Extract keywords manually (not via LLM)
        keywords = text_processor.extract_keywords(text, top_k=3)
        
        # Combine all results
        analysis_result = {
            "summary": summary,
            "title": metadata.get("title"),
            "topics": metadata.get("topics", []),
            "sentiment": metadata.get("sentiment", "neutral"),
            "keywords": keywords
        }
        
        # Store in database
        db_analysis = create_analysis(db, text, analysis_result)
        
        logger.info(f"Analysis completed and stored with ID: {db_analysis.id}")
        
        return AnalysisResponse(
            id=db_analysis.id,
            text=text,
            summary=summary,
            title=metadata.get("title"),
            topics=metadata.get("topics", []),
            sentiment=metadata.get("sentiment", "neutral"),
            keywords=keywords,
            created_at=db_analysis.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during text analysis")

@app.get("/search", response_model=List[SearchResponse])
async def search_analyses(
    topic: Optional[str] = None,
    keyword: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Search stored analyses by topic, keyword, or sentiment.
    """
    try:
        if not topic and not keyword and not sentiment:
            raise HTTPException(
                status_code=400, 
                detail="At least one search parameter (topic, keyword, or sentiment) is required"
            )
        
        logger.info(f"Searching analyses with topic='{topic}', keyword='{keyword}', sentiment='{sentiment}'")
        
        results = search_analyses(db, topic=topic, keyword=keyword, sentiment=sentiment, limit=limit)
        
        search_results = []
        for analysis in results:
            # Calculate confidence score based on search term matches
            confidence = 0.5  # Base score
            if topic and analysis.topics:
                topic_matches = sum(1 for t in analysis.topics if topic.lower() in t.lower())
                confidence += topic_matches * 0.2
            if keyword and analysis.keywords:
                keyword_matches = sum(1 for k in analysis.keywords if keyword.lower() in k.lower())
                confidence += keyword_matches * 0.15
            if sentiment and analysis.sentiment == sentiment.lower():
                confidence += 0.3
            confidence = min(confidence, 1.0)  # Cap at 1.0
            
            search_results.append(SearchResponse(
                id=analysis.id,
                summary=analysis.summary,
                title=analysis.title,
                topics=analysis.topics or [],
                sentiment=analysis.sentiment,
                keywords=analysis.keywords or [],
                created_at=analysis.created_at,
                confidence_score=confidence
            ))
        
        logger.info(f"Found {len(search_results)} matching analyses")
        return search_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during search")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)