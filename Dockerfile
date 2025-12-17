# -------- Stage 1: Builder --------
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Upgrade pip first
RUN python -m pip install --upgrade pip

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Preload SentenceTransformer model for faster startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# -------- Stage 2: Runtime --------
FROM python:3.11-slim

WORKDIR /app

# Set HuggingFace cache directory
ENV HF_HOME=/app/.cache/huggingface

# Copy installed Python packages from builder
COPY --from=builder /usr/local /usr/local

# Copy HuggingFace cache (optional, speeds up embeddings)
COPY --from=builder /root/.cache/huggingface /app/.cache/huggingface

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
