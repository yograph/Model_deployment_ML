# Use the official slim‐buster Python 3.9 image
FROM python:3.9.6-slim-buster

# Install OS‐level libraries needed by OpenCV (incl. GLib for threading)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libglib2.0-0 \
      libgl1 \
      libsm6 \
      libxext6 \
      libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy & install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model weights explicitly into /app/model_weights
COPY model_weights /app/model_weights

# Copy your application code
COPY api.py roi_extract.py streamlit_app.py /app/

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Launch both services in one container
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port 8501 --server.headless true"]
