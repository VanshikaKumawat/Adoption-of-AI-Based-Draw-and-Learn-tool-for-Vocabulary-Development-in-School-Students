import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import random

from streamlit_drawable_canvas import st_canvas

# Import your backend modules
import preprocessing
import model
import training
import color_fill
import sketch_completion
import reinforcement
import utils
import api

# === Constants ===
BASE_DIR = 'SketchyDatabase/organized_dataset/test'

# === Color Fill Helpers ===
def get_classes(base_dir=BASE_DIR):
    if not os.path.exists(base_dir):
        st.error(f"Dataset directory not found: {base_dir}")
        return []
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return sorted(classes)

def load_random_image(class_name, base_dir=BASE_DIR):
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.exists(class_dir):
        st.error(f"Class directory not found: {class_dir}")
        return None, None
    imgs = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not imgs:
        st.error(f"No images found in class folder: {class_dir}")
        return None, None
    img_file = random.choice(imgs)
    img_path = os.path.join(class_dir, img_file)
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img_gray, img_path

def flood_fill_image(img_gray, x, y, hex_color):
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    bgr = (rgb[2], rgb[1], rgb[0])
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    h, w = img_gray.shape
    if not (0 <= x < w and 0 <= y < h):
        st.warning("Fill point out of image bounds.")
        return img_color
    mask = np.zeros((h+2, w+2), np.uint8)
    flooded = img_color.copy()
    cv2.floodFill(flooded, mask, (x, y), bgr, loDiff=(20,20,20), upDiff=(20,20,20), flags=4 | cv2.FLOODFILL_FIXED_RANGE)
    return flooded

# === Sketch Completion UI ===
def sketch_completion_interactive():
    st.info("Sketch Completion Module")

    # Example: select class and load partial sketch (your implementation may vary)
    classes = get_classes()
    if not classes:
        st.stop()
    selected_class = st.selectbox("Select Class for Sketch Completion", classes)
    img_gray, img_path = load_random_image(selected_class)
    if img_gray is None:
        st.stop()
    
    st.write(f"Partial sketch from class **{selected_class}**: `{os.path.basename(img_path)}`")
    st.image(img_gray, caption="Partial Sketch", use_column_width=True)

    # Call your sketch_completion backend function (placeholder here)
    # result_img = sketch_completion.complete_drawing_interactive(img_gray)
    # For demonstration, just show original
    st.info("Sketch completion functionality to be implemented here.")

# === Reinforcement Feedback UI ===
def reinforcement_feedback():
    st.info("Teacher Feedback and Reinforcement Learning")

    # Example: upload student sketch and prediction
    uploaded_file = st.file_uploader("Upload student sketch (PNG/JPG)", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L")
        st.image(img, caption="Uploaded Sketch", use_column_width=True)

        # Example: predict label (your model.predict)
        prediction = model.predict(img)
        st.write(f"Model Prediction: **{prediction}**")

        # Collect teacher feedback (correction)
        correct_label = st.text_input("Correct Label (if prediction is wrong)")
        if st.button("Submit Feedback"):
            if correct_label:
                reinforcement.log_feedback(uploaded_file, prediction, correct_label)
                st.success("Feedback submitted for self-learning.")
            else:
                st.warning("Please enter the correct label.")

# === Main App ===
def main():
    st.title("Draw-and-Learn AI - Complete Application")

    menu = ["Home", "Color Fill", "Sketch Completion", "Reinforcement Feedback"]
    choice = st.sidebar.selectbox("Choose Module", menu)

    if choice == "Home":
        st.write("Welcome to the Draw-and-Learn AI project! Use the sidebar to select a module.")

    elif choice == "Color Fill":
        st.header("Interactive Sketch Color Fill Tool")

        classes = get_classes()
        if not classes:
            st.stop()

        selected_class = st.selectbox("Select Sketch Class", classes)
        img_gray, img_path = load_random_image(selected_class)
        if img_gray is None:
            st.stop()

        st.write(f"Random image from class **{selected_class}**: `{os.path.basename(img_path)}`")

        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        fill_color = st.color_picker("Pick Fill Color", "#FF0000")

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=1,
            background_image=Image.fromarray(img_rgb),
            update_streamlit=True,
            height=img_gray.shape[0],
            width=img_gray.shape[1],
            drawing_mode="point",
            key="color_fill_canvas"
        )

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if objects:
                last_obj = objects[-1]
                x = int(last_obj["left"])
                y = int(last_obj["top"])
                st.write(f"Flood fill at: (X={x}, Y={y}) with color {fill_color}")
                flooded_img = flood_fill_image(img_gray, x, y, fill_color)
                st.image(cv2.cvtColor(flooded_img, cv2.COLOR_BGR2RGB), caption="Color Filled Sketch", use_column_width=True)

    elif choice == "Sketch Completion":
        st.header("Complete-the-Drawing Tasks")
        sketch_completion_interactive()

    elif choice == "Reinforcement Feedback":
        st.header("Teacher Feedback and Reinforcement Learning")
        reinforcement_feedback()

if __name__ == "__main__":
    main()
