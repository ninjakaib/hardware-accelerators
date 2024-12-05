# Use a Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for Python packages and PyRTL visualization
RUN apt-get update && apt-get install -y \
    build-essential \
    graphviz \
    python3-graphviz \
    && rm -rf /var/lib/apt/lists/*

# Create output directory
RUN mkdir output

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./

# Create volume for output
VOLUME ["/app/output"]

# Set permissions for output directory
RUN chmod 777 /app/output

# Create a shell script to run both Python scripts
RUN echo '#!/bin/bash\n\
cd output_generators\n\
python bf16_multiplier.py\n\
cd ..\n\
python3 -m output_generators.lmul_heatmap\n\
python -m output_generators.make_svgs\n\
' > run_scripts.sh

RUN chmod +x run_scripts.sh

# Run the script
CMD ["./run_scripts.sh"]