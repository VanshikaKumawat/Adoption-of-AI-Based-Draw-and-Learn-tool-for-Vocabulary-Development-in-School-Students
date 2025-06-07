import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image

# --- Page config ---
st.set_page_config(page_title="Draw and Learn AI", layout="centered")

st.title("Draw and Learn AI: Color-Fill & Sketch Completion")

# --- Canvas for drawing ---
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Transparent orange fill
    stroke_width=3,
    stroke_color="#000000",
    background_color="#fff",
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)

# Button placeholders
col1, col2 = st.columns(2)

with col1:
    if st.button("Color-Fill"):
        if canvas_result.image_data is not None:
            # Convert canvas image to grayscale for demo
            img = np.array(canvas_result.image_data.convert('RGB'))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Dummy color-fill effect: threshold and color-fill inside shapes
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            color_fill = cv2.bitwise_and(img, img, mask=thresh)
            
            st.image(color_fill, caption="Color-Fill Result", use_column_width=True)
        else:
            st.warning("Please draw something first.")

with col2:
    if st.button("Complete the Drawing"):
        if canvas_result.image_data is not None:
            # Dummy completion: just echo the input for now
            st.image(canvas_result.image_data, caption="Completed Drawing (Dummy)", use_column_width=True)
        else:
            st.warning("Please draw something first.")

# --- Model loading (placeholder) ---
st.sidebar.header("Model Status")
st.sidebar.text("Models will be loaded here")

# TODO: Load your trained models and add real AI logic

