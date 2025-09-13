"""
LLM Knowledge Extractor: Prototype that takes in text and uses an LLM to produce both a summary and structured data
Author: Philip Ragan
CRUD operations for Analysis model
"""

from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from typing import List, Optional, Dict, Any
from datetime import datetime
from database import Analysis
import logging

logger = logging.getLogger(__name__)

# CRUD operations for Analysis model
def create_analysis(
    db: Session, 
    text: str, 
    analysis_result: Dict[str, Any]
) -> Analysis:
    """
    Create a new analysis record in the database
    """
    try:
        db_analysis = Analysis(
            text=text,
            summary=analysis_result["summary"],
            title=analysis_result.get("title"),
            topics=analysis_result.get("topics", []),
            sentiment=analysis_result["sentiment"],
            keywords=analysis_result.get("keywords", [])
        )
        
        db.add(db_analysis)
        db.commit()
        db.refresh(db_analysis)
        
        logger.info(f"Created analysis with ID: {db_analysis.id}")
        return db_analysis
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating analysis: {str(e)}")
        raise

# Retrieve analysis by ID
def get_analysis_by_id(db: Session, analysis_id: int) -> Optional[Analysis]:
    """
    Retrieve a single analysis by ID
    """
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        return analysis
    except Exception as e:
        logger.error(f"Error retrieving analysis {analysis_id}: {str(e)}")
        return None

# Search analyses by topic, keyword, or sentiment
def search_analyses(
    db: Session,
    topic: Optional[str] = None,
    keyword: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = 10
) -> List[Analysis]:
    """
    Search analyses based on topic, keyword, or sentiment
    """
    try:
        query = db.query(Analysis)
        conditions = []
        
        # Search by topic (case-insensitive, partial match)
        if topic:
            topic_lower = topic.lower()
            # SQLite JSON search for topics array
            query = query.filter(
                Analysis.topics.op('regexp')(f'(?i).*{topic_lower}.*') |
                Analysis.title.ilike(f'%{topic}%') |
                Analysis.summary.ilike(f'%{topic}%')
            )
        
        # Search by keyword (case-insensitive, partial match)
        if keyword:
            keyword_lower = keyword.lower()
            # Search in keywords JSON array and text content
            query = query.filter(
                Analysis.keywords.op('regexp')(f'(?i).*{keyword_lower}.*') |
                Analysis.text.ilike(f'%{keyword}%') |
                Analysis.summary.ilike(f'%{keyword}%')
            )
        
        # Search by sentiment (exact match)
        if sentiment:
            query = query.filter(Analysis.sentiment == sentiment.lower())
        
        # Order by creation date (most recent first) and limit results
        results = query.order_by(Analysis.created_at.desc()).limit(limit).all()
        
        logger.info(f"Search returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error searching analyses: {str(e)}")
        # Fallback to simpler search if complex query fails
        return _simple_search(db, topic, keyword, sentiment, limit)

# Simple search function as fallback when complex JSON queries fail
def _simple_search(
    db: Session,
    topic: Optional[str] = None,
    keyword: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = 10
) -> List[Analysis]:
    """
    Simplified search function as fallback when complex JSON queries fail
    """
    try:
        query = db.query(Analysis)
        
        # Simple text-based search across multiple fields
        if topic:
            query = query.filter(
                or_(
                    Analysis.title.ilike(f'%{topic}%'),
                    Analysis.summary.ilike(f'%{topic}%'),
                    Analysis.text.ilike(f'%{topic}%')
                )
            )
        
        if keyword:
            query = query.filter(
                or_(
                    Analysis.text.ilike(f'%{keyword}%'),
                    Analysis.summary.ilike(f'%{keyword}%'),
                    Analysis.title.ilike(f'%{keyword}%')
                )
            )
        
        if sentiment:
            query = query.filter(Analysis.sentiment == sentiment.lower())
        
        results = query.order_by(Analysis.created_at.desc()).limit(limit).all()
        logger.info(f"Simple search returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error in simple search: {str(e)}")
        return []

# Get all analyses
def get_all_analyses(db: Session, limit: int = 100) -> List[Analysis]:
    """
    Get all analyses (for debugging/admin purposes)
    """
    try:
        analyses = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(limit).all()
        logger.info(f"Retrieved {len(analyses)} analyses")
        return analyses
    except Exception as e:
        logger.error(f"Error retrieving all analyses: {str(e)}")
        return []

# Delete analysis by ID
def delete_analysis(db: Session, analysis_id: int) -> bool:
    """
    Delete an analysis by ID
    """
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if analysis:
            db.delete(analysis)
            db.commit()
            logger.info(f"Deleted analysis with ID: {analysis_id}")
            return True
        else:
            logger.warning(f"Analysis with ID {analysis_id} not found for deletion")
            return False
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting analysis {analysis_id}: {str(e)}")
        return False

# Update analysis
def update_analysis(db: Session, analysis_id: int, updates: Dict[str, Any]) -> Optional[Analysis]:
    """
    Update an existing analysis
    """
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            return None
        
        # Update allowed fields
        for field, value in updates.items():
            if hasattr(analysis, field) and field != 'id':
                setattr(analysis, field, value)
        
        db.commit()
        db.refresh(analysis)
        logger.info(f"Updated analysis with ID: {analysis_id}")
        return analysis
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating analysis {analysis_id}: {str(e)}")
        return None

# Get analysis statistics
def get_analysis_stats(db: Session) -> Dict[str, Any]:
    """
    Get basic statistics about stored analyses
    """
    try:
        total_count = db.query(Analysis).count()
        
        # Count by sentiment
        positive_count = db.query(Analysis).filter(Analysis.sentiment == 'positive').count()
        negative_count = db.query(Analysis).filter(Analysis.sentiment == 'negative').count()
        neutral_count = db.query(Analysis).filter(Analysis.sentiment == 'neutral').count()
        
        # Get recent activity (last 7 days)
        from datetime import datetime, timedelta
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_count = db.query(Analysis).filter(Analysis.created_at >= seven_days_ago).count()
        
        # Most common topics and keywords (simplified approach)
        analyses_with_topics = db.query(Analysis).filter(Analysis.topics.isnot(None)).limit(100).all()
        all_topics = []
        all_keywords = []
        
        for analysis in analyses_with_topics:
            if analysis.topics:
                all_topics.extend(analysis.topics)
            if analysis.keywords:
                all_keywords.extend(analysis.keywords)
        
        from collections import Counter
        top_topics = Counter(all_topics).most_common(5)
        top_keywords = Counter(all_keywords).most_common(5)
        
        return {
            "total_analyses": total_count,
            "recent_analyses": recent_count,
            "sentiment_distribution": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count
            },
            "top_topics": [{"topic": topic, "count": count} for topic, count in top_topics],
            "top_keywords": [{"keyword": keyword, "count": count} for keyword, count in top_keywords]
        }
        
    except Exception as e:
        logger.error(f"Error getting analysis stats: {str(e)}")
        return {
            "total_analyses": 0,
            "recent_analyses": 0,
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
            "top_topics": [],
            "top_keywords": []
        }

# Search analyses by date range
def search_analyses_by_date_range(
    db: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 10
) -> List[Analysis]:
    """
    Search analyses by date range
    """
    try:
        query = db.query(Analysis)
        
        if start_date:
            query = query.filter(Analysis.created_at >= start_date)
        
        if end_date:
            query = query.filter(Analysis.created_at <= end_date)
        
        results = query.order_by(Analysis.created_at.desc()).limit(limit).all()
        logger.info(f"Date range search returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Error in date range search: {str(e)}")
        return []