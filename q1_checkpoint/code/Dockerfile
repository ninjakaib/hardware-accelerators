# Use a Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Create output directory
RUN mkdir output

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY float8.py .
COPY lmul.py .
COPY lmul_hardware.py .
COPY lmul_script.py .

# Create volume for output
VOLUME ["/app/output"]

# Run the script
CMD ["python", "lmul_script.py"]