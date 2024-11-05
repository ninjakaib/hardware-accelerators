# Use a Python base image
FROM python:3.9-slim

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

# Install PyRTL and other necessary packages
RUN pip install --no-cache-dir \
    pyrtl \
    ipython

# Copy all source files
COPY float8.py .
COPY lmul.py .
COPY lmul_hardware.py .
COPY hardware_script.py .
COPY lmul_script.py .

# Create volume for output
VOLUME ["/app/output"]

# Set permissions for output directory
RUN chmod 777 /app/output

# Create a shell script to run both Python scripts
RUN echo '#!/bin/bash\n\
python hardware_script.py\n\
python lmul_script.py\n\
' > run_scripts.sh

RUN chmod +x run_scripts.sh

# Run both scripts
CMD ["./run_scripts.sh"]