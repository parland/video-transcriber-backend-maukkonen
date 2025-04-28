# Use a more flexible base image that works on multiple architectures
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including ffmpeg and build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies with increased timeout
COPY backend/requirements.txt backend/requirements-ml.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt --timeout 100
RUN pip3 install --no-cache-dir -r requirements-ml.txt --timeout 300 --retries 5

# Copy application code
COPY backend/ .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8765

# Start the application
CMD ["python3", "app.py"]