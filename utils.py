import os
import shutil
import random
import csv
from collections import defaultdict
import pandas as pd
from PIL import Image

# 1. Safe directory creation
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 2. Copy directory tree safely (avoid overwrite errors)
def copy_tree(src, dst):
    if os.path.exists(dst):
        print(f"⚠️ Destination folder '{dst}' already exists. Skipping copy.")
        return
    shutil.copytree(src, dst)

# 3. List images from a folder (filter by common extensions)
def list_images(folder):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    return [f for f in os.listdir(folder) if f.lower().endswith(exts)]

# 4. Read CSV with optional headers
def read_csv(filepath, headers=None):
    import csv
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        if headers:
            next(reader)  # skip header
        data = [row for row in reader]
    return data

# 5. Write to CSV (append mode) with optional header writing
def write_csv_row(filepath, row, write_header=False, header=None):
    import csv
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header and not file_exists and header is not None:
            writer.writerow(header)
        writer.writerow(row)

# 6. Load image with PIL and convert to RGB
def load_image(path):
    return Image.open(path).convert('RGB')

# 7. Get class distribution from folder (class -> image count)
def get_class_distribution(data_dir):
    distribution = defaultdict(int)
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            distribution[cls] = len(list_images(cls_path))
    return dict(distribution)

# 8. Random choice with seed (for reproducibility)
def random_choice(seq, seed=None):
    if seed is not None:
        random.seed(seed)
    return random.choice(seq)

# 9. Safely copy a file (create dirs if needed)
def safe_copy_file(src, dst):
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)

# 10. Load CSV to pandas dataframe (safe check)
def load_csv_df(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame()

