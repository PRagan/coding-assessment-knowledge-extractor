# LLM Knowledge Extractor

A FastAPI-based prototype appliation that analyzes unstructured text using Large Language Models (LLMs) to generate summaries and extract structured metadata.

## Features

### Core Features
- **Text Analysis**: Process unstructured text input (articles, blog posts, logs)
- **LLM Integration**: Uses OpenAI's API to generate 1-2 sentence summaries and extract metadata
- **Structured Metadata Extraction**:
  - Title (if available or inferable)
  - 3 key topics
  - Sentiment analysis (positive/neutral/negative)
  - 3 most frequent nouns (extracted manually, not via LLM)
- **Persistent Storage**: SQLite database for storing all analyses
- **REST API**:
  - `POST /analyze` - Process new text and return results
  - `GET /search` - Search stored analyses by topic, keyword, or sentiment
- **Robust Error Handling**: Handles empty input and LLM API failures gracefully

### Bonus Features
- **Docker Support**: Fully containerized with Docker and Docker Compose
- **Comprehensive Tests**: Unit and integration tests with pytest
- **Confidence Scoring**: Naive heuristic-based confidence scores
- **Health Checks**: API health monitoring
- **Fallback Functionality**: Works even without OpenAI API key (mock responses)

## Quick Start

### Option 1: Docker (Recommended)

1. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd llm-knowledge-extractor
   cp .env.example .env
   ```
   See .env.example

2. **Add your OpenAI API key** to `.env`:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

4. **Test the API**:
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Analyze text
   curl -X POST "http://localhost:8000/analyze" \
        -H "Content-Type: application/json" \
        -d '{"text": "Artificial intelligence is transforming healthcare through machine learning algorithms."}'
   
   # Search analyses
   curl "http://localhost:8000/search?topic=artificial"
   ```

### Option 2: Local Development

1. **Setup Python environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   export DATABASE_URL=sqlite:///./analyses.db
   ```

3. **Run the application**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

## API Usage

### Analyze Text
```bash
POST /analyze
Content-Type: application/json

{
  "text": "Your unstructured text here..."
}
```

**Response:**
```json
{
  "id": 1,
  "text": "Your input text...",
  "summary": "Generated 1-2 sentence summary.",
  "title": "Extracted or inferred title",
  "topics": ["topic1", "topic2", "topic3"],
  "sentiment": "positive",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "created_at": "2024-01-01T12:00:00"
}
```

### Search Analyses
```bash
GET /search?topic=ai&sentiment=positive&limit=10
```

**Response:**
```json
[
  {
    "id": 1,
    "summary": "Analysis summary...",
    "title": "Title if available",
    "topics": ["ai", "technology", "innovation"],
    "sentiment": "positive",
    "keywords": ["artificial", "intelligence", "technology"],
    "created_at": "2024-01-01T12:00:00"
  }
]
```

## Design Choices

**Architecture**: FastAPI was chosen for rapid development and excellent automatic API documentation. The modular structure separates concerns between API handling, business logic (LLM integration), and data persistence.

**Database**: SQLite provides zero-configuration persistence suitable for this prototype scale, with easy migration path to PostgreSQL for production use.

**Error Handling**: Implemented fallback responses when OpenAI API is unavailable, ensuring the service remains functional even during external service outages.

**Manual Keyword Extraction**: Used NLTK for POS tagging to identify nouns, with regex fallback when NLTK data is unavailable, fulfilling the requirement to implement this feature independently of LLM services.

**Testing Strategy**: Comprehensive test suite covers API endpoints, text processing, database operations, and integration workflows to ensure reliability.

## Trade-offs

Due to time constraints (90-minute target), several trade-offs were made:

- **Simple Search**: Basic string matching for search functionality rather than advanced text similarity
- **Basic Confidence Scoring**: Naive heuristic-based confidence rather than sophisticated ML-based scoring  
- **Limited Batch Processing**: Basic implementation without advanced queuing or async processing
- **Minimal UI**: Focused on API implementation rather than web interface
- **Basic Logging**: Simple logging setup rather than comprehensive monitoring

## Production Considerations

For production deployment, consider:
- Replace SQLite with PostgreSQL or similar production database
- Add authentication and rate limiting
- Implement proper secrets management
- Add comprehensive monitoring and alerting
- Scale with load balancers and multiple instances
- Enhance error handling and retry mechanisms
- Add data validation and sanitization

## API Documentation

When running, visit `http://localhost:8000/docs` for interactive Swagger API documentation.

## License

This project was originally created for the Jouster take-home assignment.