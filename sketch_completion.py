# sketch_completion.py

import os
import random
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

# === Constants / Paths ===
FULL_IMG_DIR = 'SketchyDatabase/organized_dataset/test'
PARTIAL_IMG_DIR = 'partial_sketches'


# --- 1. Generate Partial Sketches (if needed) ---
def generate_partial_image(img, method='bottom_half'):
    h, w = img.shape
    mask = np.ones((h, w), dtype=np.uint8) * 255
    if method == 'bottom_half':
        mask[h // 2:, :] = 0
    elif method == 'left_half':
        mask[:, :w // 2] = 0
    elif method == 'random_patch':
        x, y = random.randint(0, w // 2), random.randint(0, h // 2)
        mask[y:y + h // 3, x:x + w // 3] = 0
    return cv2.bitwise_and(img, img, mask=mask)


def generate_partial_sketches(full_dir=FULL_IMG_DIR, partial_dir=PARTIAL_IMG_DIR):
    os.makedirs(partial_dir, exist_ok=True)
    for class_folder in os.listdir(full_dir):
        class_path = os.path.join(full_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        save_class_dir = os.path.join(partial_dir, class_folder)
        os.makedirs(save_class_dir, exist_ok=True)

        for fname in os.listdir(class_path):
            if not fname.endswith('.png'):
                continue
            full_path = os.path.join(class_path, fname)
            img = cv2.imread(full_path, 0)
            method = random.choice(['bottom_half', 'left_half', 'random_patch'])
            partial = generate_partial_image(img, method)
            save_path = os.path.join(save_class_dir, fname)
            cv2.imwrite(save_path, partial)
    st.success(f"✅ Partial sketches saved in: {partial_dir}")


# --- 2. Feature Extractor ---
class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = torch.nn.Sequential(*list(vgg.features.children())[:30])  # conv5_3 layer
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, image: Image.Image):
        image = self.transforms(image).unsqueeze(0)
        with torch.no_grad():
            features = self.features(image)
        return features.view(-1)


# --- 3. Shape matching ---
def match_shapes(img1, img2):
    cnts1, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts1 and cnts2:
        return cv2.matchShapes(cnts1[0], cnts2[0], 1, 0.0)
    return 1.0  # worst match if contours not found


# --- 4. Evaluate completion ---
def evaluate_completion(partial_path, completed_path, true_path):
    st.write("📄 Evaluating drawing...")

    # Load grayscale images
    img_partial = cv2.imread(partial_path, 0)
    img_completed = cv2.imread(completed_path, 0)
    img_true = cv2.imread(true_path, 0)

    # Shape similarity
    shape_score = match_shapes(img_completed, img_true)

    # Embedding similarity
    extractor = VGGFeatureExtractor()
    img_completed_rgb = Image.open(completed_path).convert("RGB")
    img_true_rgb = Image.open(true_path).convert("RGB")
    emb_completed = extractor(img_completed_rgb)
    emb_true = extractor(img_true_rgb)
    cos_sim = cosine_similarity(emb_completed.view(1, -1), emb_true.view(1, -1))[0][0]

    st.write(f"🔍 Shape Match Score (lower is better): {shape_score:.4f}")
    st.write(f"🔍 Embedding Cosine Similarity (higher is better): {cos_sim:.4f}")

    if shape_score < 0.25 and cos_sim > 0.85:
        st.success("✅ Drawing Completed Correctly!")
    else:
        st.error("❌ Drawing Incomplete or Incorrect.")


# --- 5. Display Before and After images ---
def display_before_after(partial_path, completed_path):
    img_partial = cv2.imread(partial_path, 0)
    img_completed = cv2.imread(completed_path, 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_partial, cmap='gray')
    axes[0].set_title("Partial Sketch")
    axes[0].axis('off')

    axes[1].imshow(img_completed, cmap='gray')
    axes[1].set_title("Completed Sketch")
    axes[1].axis('off')

    st.pyplot(fig)


# --- 6. Streamlit UI ---
def streamlit_app():
    st.title("Sketch Completion Evaluation")

    # Optional: Button to generate partial sketches (only run once or as needed)
    if st.button("Generate Partial Sketches"):
        generate_partial_sketches()

    # Select class
    if not os.path.exists(PARTIAL_IMG_DIR):
        st.warning(f"Partial sketches folder not found: {PARTIAL_IMG_DIR}. Generate partial sketches first.")
        return

    class_names = sorted([d for d in os.listdir(PARTIAL_IMG_DIR) if os.path.isdir(os.path.join(PARTIAL_IMG_DIR, d))])
    selected_class = st.selectbox("Select Class", class_names)

    if selected_class:
        partial_class_dir = os.path.join(PARTIAL_IMG_DIR, selected_class)
        full_class_dir = os.path.join(FULL_IMG_DIR, selected_class)

        partial_files = sorted([f for f in os.listdir(partial_class_dir) if f.endswith('.png')])
        selected_file = st.selectbox("Select Partial Sketch", partial_files)

        if selected_file:
            partial_path = os.path.join(partial_class_dir, selected_file)
            completed_path = os.path.join(full_class_dir, selected_file)
            true_path = completed_path

            if os.path.exists(partial_path) and os.path.exists(completed_path):
                display_before_after(partial_path, completed_path)
                evaluate_completion(partial_path, completed_path, true_path)
            else:
                st.error("Selected file paths do not exist.")


if __name__ == "__main__":
    streamlit_app()
