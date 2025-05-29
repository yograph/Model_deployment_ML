# Use the slim-buster Python image
FROM python:3.9.6-slim-buster

# Install OS-level libraries for OpenCV (incl. the GLib thread lib)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libglib2.0-0 \
      libgl1 \
      libsm6 \
      libxext6 \
      libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy & install all Python deps (including uvicorn & streamlit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code and model weights
COPY api.py roi_extract.py streamlit_app.py model_weights/ /app/

# Expose FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Launch both services (exec form)
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port 8501 --server.headless true"]
