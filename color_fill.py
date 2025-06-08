import os
import random
import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# === Constants ===
BASE_DIR = 'SketchyDatabase/organized_dataset/test'

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
    # Convert hex to BGR tuple
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

def main():
    st.title("Sketch Color Fill Tool - Click to Fill")

    classes = get_classes()
    if not classes:
        st.stop()

    selected_class = st.selectbox("Select Sketch Class", classes)
    img_gray, img_path = load_random_image(selected_class)
    if img_gray is None:
        st.stop()

    st.write(f"Random image from class **{selected_class}**: `{os.path.basename(img_path)}`")

    # Convert grayscale to RGB for displaying and canvas background
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    fill_color = st.color_picker("Pick Fill Color", "#FF0000")

    # Canvas settings
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",  # Transparent drawing color (we don't draw here)
        stroke_width=1,
        background_image=Image.fromarray(img_rgb),
        update_streamlit=True,
        height=img_gray.shape[0],
        width=img_gray.shape[1],
        drawing_mode="point",  # lets user click points
        key="color_fill_canvas"
    )

    if canvas_result.json_data is not None:
        # Extract click coordinates from canvas events
        objects = canvas_result.json_data["objects"]
        if objects:
            # Take the last clicked point
            last_obj = objects[-1]
            x = int(last_obj["left"])
            y = int(last_obj["top"])

            st.write(f"Flood fill at: (X={x}, Y={y}) with color {fill_color}")

            flooded_img = flood_fill_image(img_gray, x, y, fill_color)
            st.image(cv2.cvtColor(flooded_img, cv2.COLOR_BGR2RGB), caption="Color Filled Sketch", use_column_width=True)

if __name__ == "__main__":
    main()
