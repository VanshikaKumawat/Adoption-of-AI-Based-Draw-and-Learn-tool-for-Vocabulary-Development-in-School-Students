# backend/preprocessing.py

import os
import shutil
import random
from PIL import Image

def resize_and_convert(input_dir, output_dir, size=(256, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for filename in os.listdir(class_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                img_path = os.path.join(class_path, filename)
                img = Image.open(img_path).convert("L")
                img = img.resize(size)
                img.save(os.path.join(output_class_dir, filename))

def split_dataset(input_dir, output_train, output_test, split_ratio=0.8):
    if not os.path.exists(output_train): os.makedirs(output_train)
    if not os.path.exists(output_test): os.makedirs(output_test)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path): continue

        images = os.listdir(class_path)
        random.shuffle(images)

        split_idx = int(split_ratio * len(images))
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        os.makedirs(os.path.join(output_train, class_name), exist_ok=True)
        os.makedirs(os.path.join(output_test, class_name), exist_ok=True)

        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(output_train, class_name, img))

        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(output_test, class_name, img))

def get_class_labels(input_dir):
    return sorted(os.listdir(input_dir))
