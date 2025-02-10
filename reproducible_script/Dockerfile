# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Note: This Dockerfile must be built from the root directory of the project
# Build command: docker build -f reproducible_script/Dockerfile .

# Copy requirements and test files
COPY reproducible_script/requirements.txt .
COPY reproducible_script/run_tests.py .
COPY tests/ ./tests/
COPY hardware_accelerators/ ./hardware_accelerators/
COPY rtllib/ ./rtllib/
COPY data/ ./data/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Command to run tests
CMD ["python", "run_tests.py"] 