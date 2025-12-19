# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# Dockerfile for CORD-19 Search Engine on Hugging Face Spaces

FROM python:3.9-slim

# Create non-root user for security (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Download spaCy language model (for text processing)
RUN python -m spacy download en_core_web_sm

# Copy all application code and data
COPY --chown=user . /app

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Set environment variables
ENV FLASK_APP=src/app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run Flask with gunicorn for production
# Note: app:app means module 'app' and Flask instance 'app'
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "--chdir", "src", "app:app"]
