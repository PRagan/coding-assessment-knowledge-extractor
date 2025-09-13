import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import tempfile
import json

# Import application components
from app.main import app
from app.database import Base, get_db, Analysis
from app.services.text_processor import TextProcessor
from app.services.llm_service import LLMService

# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

# Override database dependency
app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)

@pytest.fixture(scope="function")
def setup_database():
    """Create test database tables"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

class TestAPI:
    """Test API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "LLM Knowledge Extractor API" in response.json()["message"]
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_analyze_valid_text(self, setup_database):
        """Test text analysis with valid input"""
        test_text = "This is a test article about artificial intelligence and machine learning. AI is transforming many industries today."
        
        response = client.post(
            "/analyze",
            json={"text": test_text}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "id" in data
        assert "summary" in data
        assert "sentiment" in data
        assert "keywords" in data
        assert "topics" in data
        assert data["text"] == test_text
        assert data["sentiment"] in ["positive", "negative", "neutral"]
        assert isinstance(data["keywords"], list)
        assert isinstance(data["topics"], list)
    
    def test_analyze_empty_text(self, setup_database):
        """Test analysis with empty text"""
        response = client.post(
            "/analyze",
            json={"text": ""}
        )
        
        assert response.status_code == 400
        assert "Empty text input" in response.json()["detail"]
    
    def test_analyze_whitespace_only(self, setup_database):
        """Test analysis with whitespace-only text"""
        response = client.post(
            "/analyze",
            json={"text": "   \n\t   "}
        )
        
        assert response.status_code == 400
        assert "Empty text input" in response.json()["detail"]
    
    def test_search_without_parameters(self, setup_database):
        """Test search without any parameters"""
        response = client.get("/search")
        
        assert response.status_code == 400
        assert "at least one search parameter" in response.json()["detail"].lower()
    
    def test_search_by_topic(self, setup_database):
        """Test search by topic"""
        # First, create some test data
        test_text = "This article discusses climate change and global warming."
        client.post("/analyze", json={"text": test_text})
        
        # Search by topic
        response = client.get("/search?topic=climate")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_search_by_sentiment(self, setup_database):
        """Test search by sentiment"""
        # Create test data
        test_text = "This is wonderful news about technological advancement!"
        client.post("/analyze", json={"text": test_text})
        
        # Search by sentiment
        response = client.get("/search?sentiment=positive")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

class TestTextProcessor:
    """Test text processing functionality"""
    
    def setUp(self):
        self.processor = TextProcessor()
    
    def test_keyword_extraction_basic(self):
        """Test basic keyword extraction"""
        processor = TextProcessor()
        text = "The cat sat on the mat. The cat was very comfortable on the soft mat."
        
        keywords = processor.extract_keywords(text, top_k=3)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 3
        # Should find 'cat' and 'mat' as they appear multiple times
        assert any('cat' in word.lower() for word in keywords) or any('mat' in word.lower() for word in keywords)
    
    def test_keyword_extraction_empty_text(self):
        """Test keyword extraction with empty text"""
        processor = TextProcessor()
        keywords = processor.extract_keywords("", top_k=3)
        
        assert keywords == []
    
    def test_keyword_extraction_stopwords_filtered(self):
        """Test that stop words are filtered out"""
        processor = TextProcessor()
        text = "The the the and and or but in on at to for of with by"
        
        keywords = processor.extract_keywords(text, top_k=5)
        
        # Should return empty list or very few words since these are all stop words
        assert len(keywords) <= 1
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation"""
        processor = TextProcessor()
        
        text = "This is a substantial article about artificial intelligence and machine learning with plenty of content."
        summary = "This article discusses AI and ML technologies."
        keywords = ["artificial", "intelligence", "machine"]
        
        score = processor.calculate_confidence_score(text, summary, keywords)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should have decent confidence with good inputs
    
    def test_confidence_score_low_quality(self):
        """Test confidence score with low quality inputs"""
        processor = TextProcessor()
        
        text = "Short."
        summary = "Summary generated offline"
        keywords = []
        
        score = processor.calculate_confidence_score(text, summary, keywords)
        
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should have lower confidence

class TestLLMService:
    """Test LLM service functionality"""
    
    @pytest.mark.asyncio
    async def test_mock_summary_when_no_api_key(self):
        """Test that mock summary is generated when no API key is provided"""
        # Temporarily remove API key
        original_key = os.getenv("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        service = LLMService()
        text = "This is a test text for summary generation."
        
        summary = await service.generate_summary(text)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        
        # Restore API key
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
    
    @pytest.mark.asyncio
    async def test_mock_metadata_when_no_api_key(self):
        """Test that mock metadata is generated when no API key is provided"""
        # Temporarily remove API key
        original_key = os.getenv("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        service = LLMService()
        text = "This is a test text for metadata extraction."
        
        metadata = await service.extract_metadata(text)
        
        assert isinstance(metadata, dict)
        assert "title" in metadata
        assert "topics" in metadata
        assert "sentiment" in metadata
        assert isinstance(metadata["topics"], list)
        assert len(metadata["topics"]) == 3
        assert metadata["sentiment"] in ["positive", "negative", "neutral"]
        
        # Restore API key
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
    
    def test_metadata_validation(self):
        """Test metadata validation and cleaning"""
        service = LLMService()
        
        # Test with invalid metadata
        invalid_metadata = {
            "title": "",
            "topics": ["topic1"],  # Less than 3
            "sentiment": "very_positive"  # Invalid sentiment
        }
        
        cleaned = service._validate_metadata(invalid_metadata)
        
        assert cleaned["title"] is None  # Empty string should become None
        assert len(cleaned["topics"]) == 3  # Should be padded to 3
        assert cleaned["sentiment"] == "neutral"  # Invalid should become neutral

class TestDatabase:
    """Test database operations"""
    
    def test_database_connection(self, setup_database):
        """Test database connection and table creation"""
        db = TestingSessionLocal()
        try:
            # Try to query the analyses table
            count = db.query(Analysis).count()
            assert count == 0  # Should be empty initially
        finally:
            db.close()

class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_complete_analysis_workflow(self, setup_database):
        """Test complete analysis and search workflow"""
        # Step 1: Analyze text
        test_text = "Artificial intelligence is revolutionizing healthcare through machine learning algorithms and data analysis."
        
        response1 = client.post("/analyze", json={"text": test_text})
        assert response1.status_code == 200
        analysis1 = response1.json()
        
        # Step 2: Analyze another text
        test_text2 = "Climate change poses significant challenges to global agriculture and food security."
        
        response2 = client.post("/analyze", json={"text": test_text2})
        assert response2.status_code == 200
        analysis2 = response2.json()
        
        # Step 3: Search by keyword
        search_response = client.get("/search?keyword=intelligence")
        assert search_response.status_code == 200
        results = search_response.json()
        
        # Should find the first analysis
        assert len(results) >= 1
        
        # Step 4: Search by topic
        search_response2 = client.get("/search?topic=climate")
        assert search_response2.status_code == 200
        results2 = search_response2.json()
        
        # Should find the second analysis
        assert len(results2) >= 1

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])