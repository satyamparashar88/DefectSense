FROM python:3.13-slim

# Application directory
WORKDIR /app

# Python environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install PyTorch CPU packages
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the DefectSense application
COPY . .

# Backend and frontend ports
EXPOSE 8000
EXPOSE 8501