# ===== File: models/main_model/roi_extract.py =====
#!/usr/bin/env python3
"""
ROI extraction utilities: Otsu, adaptive threshold, Canny.
Place at `models/main_model/roi_extract.py`.
"""
import cv2
import numpy as np

def extract_roi_otsu(gray: np.ndarray, gkernel=(5, 5)):
    h, w = gray.shape
    up = np.percentile(gray, 95)
    gray_clipped = np.where(gray > up, gray.min(), gray).astype(np.uint8)
    if gkernel:
        gray_clipped = cv2.GaussianBlur(gray_clipped, gkernel, 0)
    _, bw = cv2.threshold(gray_clipped, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.dilate(bw, kern, iterations=1)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    area_pct = area / (h*w)
    x,y,ww,hh = cv2.boundingRect(c)
    return (x, y, x+ww, y+hh), area_pct

def extract_roi_adaptive(gray: np.ndarray, block_size=51, C=5):
    bw = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,
                               block_size, C)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kern)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    h, w = gray.shape
    area = cv2.contourArea(c)
    area_pct = area / (h*w)
    x,y,ww,hh = cv2.boundingRect(c)
    return (x, y, x+ww, y+hh), area_pct

def extract_roi_canny(gray: np.ndarray, low=50, high=150):
    edges = cv2.Canny(gray, low, high)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dil = cv2.dilate(edges, kern, iterations=1)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    h, w = gray.shape
    area = cv2.contourArea(c)
    area_pct = area / (h*w)
    x,y,ww,hh = cv2.boundingRect(c)
    return (x, y, x+ww, y+hh), area_pct
