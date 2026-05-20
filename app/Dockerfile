# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY .env .env

# Metadata for GHCR
LABEL org.opencontainers.image.source=https://github.com/nehruprakashb123/gitdemo
LABEL org.opencontainers.image.description="DevOps AI Copilot"

# Expose port
EXPOSE 8000

# Run the application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]