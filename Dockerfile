# Use official PyTorch CPU base image for lightweight deployment
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

# Install system dependencies (OpenCV/GL support for YOLO & PIL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install Python packages (using PyTorch CPU wheel to save image size and RAM)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (7860 for Hugging Face Spaces / 8000 for standard FastAPI)
EXPOSE 7860

# Default command: launch Streamlit interface
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
