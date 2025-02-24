# Dockerfile for the demo
FROM python:3.12-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY demo/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and application files
COPY models/ /code/models/
COPY hardware_accelerators/ /code/hardware_accelerators/
COPY demo/app.py /code/app.py

# Set environment variables for Gradio
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Expose the port Gradio runs on
EXPOSE 7860

# Command to run the Gradio app
CMD ["python", "app.py"]