#!/usr/bin/env python3
"""
Simple Streamlit front-end to upload an image,
send it to the FastAPI `/predict` endpoint, and display the results.
"""
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import numpy as np

# Base URL for local FastAPI
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="ROI Classifier Demo", layout="centered")

st.title("ROI Classifier Demo")
st.write("Upload an image, see ROI-method, bbox, and model probabilities.")

uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp","gif","tiff"])

if uploaded:
    # Display the uploaded image
    img = Image.open(uploaded)
    st.image(img, caption="Input Image", use_column_width=True)

    # Convert to bytes and POST
    if st.button("Classify"):
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        with st.spinner("Sending to API..."):
            resp = requests.post(API_URL, files=files)
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Method: **{data['roi_method']}**")
            if data["bbox"]:
                x0,y0,x1,y1 = data["bbox"]
                st.write(f"Bounding box: {data['bbox']}")
                # show cropped ROI
                arr = np.array(img)
                roi = arr[y0:y1, x0:x1]
                st.image(roi, caption="Extracted ROI", use_column_width=True)
            st.subheader("Classifier Probabilities")
            for p in data["predictions"]:
                st.write(f"- **Model {p['classifier']}**: {p['probs']}")
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
