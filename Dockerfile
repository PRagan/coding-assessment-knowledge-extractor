# --- Builder stage ---
FROM python:3.12-slim AS builder

# Update, upgrade, and install Linux packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
ENV NLTK_DATA=/usr/local/nltk_data
RUN mkdir -p $NLTK_DATA && chmod -R 777 $NLTK_DATA

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader -d $NLTK_DATA punkt stopwords averaged_perceptron_tagger

# --- Final stage ---
FROM python:3.12-slim

# Update, upgrade, and install Linux packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the application
RUN groupadd -r appuser && useradd -m -r -g appuser appuser

# Set environment variables
ENV NLTK_DATA=/usr/local/nltk_data

# Ensure NLTK data directory exists and is writable
RUN mkdir -p $NLTK_DATA && chmod -R 777 $NLTK_DATA

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY . .

# Set working directory
WORKDIR /app

# Ensure __init__.py files exist for package recognition
RUN touch /app/__init__.py && touch /app/services/__init__.py

# Ensure /app/data exists and is owned by appuser
RUN mkdir -p /app/data && chown -R appuser:appuser /app

# Change to non-root user
USER appuser

# Set environment variables for the application
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV DATABASE_URL=sqlite:////app/data/analyses.db

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
