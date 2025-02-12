FROM mcr.microsoft.com/devcontainers/python:1-3.12

# Install required packages including graphviz
RUN sudo apt-get update && \
    sudo apt-get install -y \
    graphviz \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp