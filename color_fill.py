import os
import random
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# === Constants ===
BASE_DIR = 'SketchyDatabase/organized_dataset/test'

def get_classes(base_dir=BASE_DIR):
    """Return sorted list of class folders."""
    if not os.path.exists(base_dir):
        st.error(f"Dataset directory not found: {base_dir}")
        return []
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return sorted(classes)

def load_random_image(class_name, base_dir=BASE_DIR):
    """Load a random grayscale image from the given class."""
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
    """Apply flood fill to grayscale sketch at (x,y) with the specified hex color."""
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

# === Streamlit App ===
def main():
    st.title("Sketch Color Fill Tool")

    classes = get_classes()
    if not classes:
        st.stop()

    selected_class = st.selectbox("Select Sketch Class", classes)
    img_gray, img_path = load_random_image(selected_class)
    if img_gray is None:
        st.stop()

    st.write(f"Randomly selected image from class **{selected_class}**: `{os.path.basename(img_path)}`")

    st.image(img_gray, caption="Original Sketch (Grayscale)", use_column_width=True)

    st.write("### Select Fill Color and Point")

    col1, col2 = st.columns([1, 2])
    with col1:
        fill_color = st.color_picker("Pick Fill Color", "#FF0000")
        x = st.number_input("X Coordinate", min_value=0, max_value=img_gray.shape[1]-1, value=img_gray.shape[1]//2)
        y = st.number_input("Y Coordinate", min_value=0, max_value=img_gray.shape[0]-1, value=img_gray.shape[0]//2)

        if st.button("Apply Color Fill"):
            flooded_img = flood_fill_image(img_gray, int(x), int(y), fill_color)
            st.image(cv2.cvtColor(flooded_img, cv2.COLOR_BGR2RGB), caption="Color Filled Sketch", use_column_width=True)

    with col2:
        st.write("**Instructions:**")
        st.markdown("""
        - Select a class to load a random sketch.
        - Pick a fill color using the color picker.
        - Enter X and Y coordinates (pixel positions) for the flood fill start point.
        - Click **Apply Color Fill** to see the result.
        """)

if __name__ == "__main__":
    main()
