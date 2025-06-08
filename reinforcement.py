import os
import random
import shutil
import csv
from collections import defaultdict
import pandas as pd
import streamlit as st
from PIL import Image

# --- CONFIGURATION ---
# Use relative or config-driven paths for better GitHub/portability
BASE_PATH = os.getenv("DRAWLEARN_BASE_PATH", "./SketchyDatabase")

FULL_DATA_PATH = os.path.join(BASE_PATH, "organized_dataset")
SUBSET_PATH = os.path.join(BASE_PATH, "DrawLearn10")
MODEL_PATH = os.path.join(BASE_PATH, "models", "vgg16_teacher_v1.pth")

FEEDBACK_LOG = os.path.join(BASE_PATH, "feedback_logs", "teacher_feedback.csv")
FEEDBACK_IMG_STORE = os.path.join(BASE_PATH, "self_learning_data")

CLASS_LIST = ['bicycle', 'apple', 'cup', 'bench', 'flower',
              'door', 'fan', 'knife', 'piano', 'airplane']

# Ensure directories exist
os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
os.makedirs(FEEDBACK_IMG_STORE, exist_ok=True)

# --- Step 1: Prepare 10-class subset (call once) ---
def make_subset(selected_classes=CLASS_LIST, full_data_path=FULL_DATA_PATH, subset_path=SUBSET_PATH):
    for split in ['train', 'test']:
        for cls in selected_classes:
            src = os.path.join(full_data_path, split, cls)
            dst = os.path.join(subset_path, split, cls)
            if os.path.exists(src):
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copytree(src, dst)
            else:
                print(f"⚠️ Class '{cls}' not found in {split} set.")
    print(f"✅ 10-class subset prepared at: {subset_path}")

# --- Step 2: Simulate AI Prediction ---
def simulate_prediction():
    predicted_class = random.choice(CLASS_LIST)
    confidence_score = round(random.uniform(0.6, 0.95), 2)  # mock confidence score
    return predicted_class, confidence_score

# --- Step 3: Log Teacher Feedback ---
def log_feedback(image_path, predicted, confidence, correct_label):
    file_exists = os.path.isfile(FEEDBACK_LOG)
    with open(FEEDBACK_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["image_path", "predicted", "confidence", "corrected"])
        writer.writerow([image_path, predicted, confidence, correct_label])
    print(f"📝 Feedback logged: {os.path.basename(image_path)} → predicted: {predicted}, corrected: {correct_label}")

# --- Step 4: Collect teacher-validated drawings for self-learning ---
def collect_corrected_drawings():
    if not os.path.isfile(FEEDBACK_LOG):
        print("⚠️ No feedback log found.")
        return
    df = pd.read_csv(FEEDBACK_LOG)
    for i, row in df.iterrows():
        corrected_cls_dir = os.path.join(FEEDBACK_IMG_STORE, row['corrected'])
        os.makedirs(corrected_cls_dir, exist_ok=True)
        new_filename = f"img_{i}_{os.path.basename(row['image_path'])}"
        try:
            shutil.copy2(row['image_path'], os.path.join(corrected_cls_dir, new_filename))
        except FileNotFoundError:
            print(f"⚠️ File not found: {row['image_path']}")
    print(f"✅ Collected corrected samples into: {FEEDBACK_IMG_STORE}")

# --- Step 5: Monitor retraining trigger ---
def check_retraining_trigger(min_samples=10):
    total = 0
    classwise = defaultdict(int)
    if not os.path.exists(FEEDBACK_IMG_STORE):
        print(f"⚠️ Self-learning data folder '{FEEDBACK_IMG_STORE}' does not exist.")
        return
    for cls in os.listdir(FEEDBACK_IMG_STORE):
        cls_path = os.path.join(FEEDBACK_IMG_STORE, cls)
        if os.path.isdir(cls_path):
            sample_count = len(os.listdir(cls_path))
            total += sample_count
            classwise[cls] = sample_count
    print(f"📦 Total corrected samples: {total}")
    for cls, count in classwise.items():
        print(f"  - {cls}: {count} samples")
    if total >= min_samples:
        print("🟡 Retraining trigger condition met — initiate model update now.")
    else:
        print(f"🔵 Waiting... Need {min_samples - total} more samples.")

# --- Step 6: Summarize feedback log ---
def summarize_feedback_log():
    if not os.path.isfile(FEEDBACK_LOG):
        print("⚠️ No feedback log found.")
        return
    df = pd.read_csv(FEEDBACK_LOG)
    st.write("## 📊 Feedback Summary")
    st.write(df['corrected'].value_counts())
    st.write("\n## 🔁 Misclassification Summary")
    mismatches = (df['predicted'] != df['corrected']).sum()
    st.write(f"  - Misclassifications: {mismatches}")
    st.write(f"  - Total entries: {len(df)}")

# --- Step 7: Streamlit visualization of feedback cycle ---
def show_feedback_cycle():
    st.title("Teacher AI Reinforcement Feedback")

    # Select a class and show sample images
    subset_test_dir = os.path.join(SUBSET_PATH, "test")
    class_selected = st.selectbox("Select class to review:", CLASS_LIST)

    class_dir = os.path.join(subset_test_dir, class_selected)
    if not os.path.exists(class_dir):
        st.warning(f"No test data for class '{class_selected}'")
        return

    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        st.warning(f"No images found in '{class_dir}'")
        return

    # Pick a random sample image
    img_file = random.choice(images)
    img_path = os.path.join(class_dir, img_file)

    # Display original sketch (partial)
    st.subheader("Partial Sketch / Original Image")
    img = Image.open(img_path)
    st.image(img, use_column_width=True)

    # Simulate prediction and show
    predicted_class, confidence = simulate_prediction()
    st.write(f"**AI Prediction:** {predicted_class} (Confidence: {confidence})")

    # Teacher correction input
    corrected_label = st.selectbox("Teacher Correction (select correct class):", CLASS_LIST, index=CLASS_LIST.index(class_selected))

    if st.button("Submit Feedback"):
        log_feedback(img_path, predicted_class, confidence, corrected_label)
        st.success("✅ Feedback submitted and logged!")

    # Show feedback log summary
    if st.checkbox("Show feedback summary and misclassification stats"):
        summarize_feedback_log()

# --- For direct run ---
if __name__ == "__main__":
    # For testing outside Streamlit
    make_subset()
    collect_corrected_drawings()
    check_retraining_trigger()
