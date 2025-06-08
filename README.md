# Adoption-of-AI-Based-Draw-and-Learn-Tool-for-Vocabulary-Development-in-School-Students

## 🚀 Project Overview

This project aims to revolutionize vocabulary learning for primary school students by integrating cutting-edge AI into a gamified drawing-based learning tool. The goal is to allow students to learn English and Telugu vocabulary through drawing, receiving real-time AI feedback, and enabling continuous model improvement via teacher input.

---

## 🎓 Educational Focus

* Target Users: Primary school students (up to Class 5) across Telangana
* Languages: English and Telugu
* Skills Enhanced: Visual vocabulary association, drawing, language comprehension, creativity

---

## 🧪 AI Functionalities

### 1. AI-Powered Drawing Recognition Engine

* **Task**: Classify student-drawn sketches into vocabulary categories (image classification)
* **Examples**:

  * Prompt: "Draw an Apple"

    * AI recognizes a circular shape with a stem and responds, "That looks like a perfect Apple!"
  * Prompt: "Draw a Car"

    * AI unsure if it's a car or bus: "Hmm, I'm not sure, does it have doors?"
  * Prompt: "Draw a Dog"

    * AI misclassifies as cat and responds: "That looks like a Cat! Can you try drawing a Dog?"
* **Mechanism**: CNN-based classifier with confidence-based feedback

### 2. Enhanced Drawing Interactions

#### a) Color-Fill Feature

* **Goal**: Let children fill drawn regions with colors using mouse clicks
* **Techniques**: Morphological operations, contour detection, flood fill (OpenCV)
* **Challenge**: Handle sketches with small open gaps to prevent color leakage

#### b) Complete-the-Drawing Tasks

* **Goal**: Students complete partial sketches provided by the system
* **Evaluation**: Shape embedding or contour similarity used to assess if the drawing is complete

### 3. AI-Based Hints

* **Purpose**: Suggest clues when the AI cannot confidently recognize a drawing
* **Examples**:

  * For incomplete sun: "Try adding some lines around the circle for rays!"
  * For a roofless house: "Houses usually have a roof on top!"
* **Approach**: Feature map analysis or rule-based feedback system

### 4. Inappropriate Content Guardrails

* **Task**: Detect and block inappropriate content or symbols
* **Approach**: Dedicated classifier or pretrained moderation models
* **Failsafe**: Allow teachers to override false positives or missed detections

---

## 🚜 Dataset

* **Primary Dataset**: [Sketchy Database](https://sketchy.eye.gatech.edu/) (\~6GB)
* **Classes Used for Testing**: A subset of 10 classes from Sketchy used for demo and prototyping
* **Organization**:

  * `/SketchyDatabase/organized_dataset/train`
  * `/SketchyDatabase/organized_dataset/test`
* **Dataset Handling**:

  * Manual upload for local testing
  * Google Drive + gdown script available for cloud download

---

## 🚀 AI Development Lifecycle

### Phase 1: Initial Training

* **Preprocessing**: Normalization, augmentation (scaling, rotation, flipping)
* **Model**: VGG16 / ResNet18 for initial classification
* **Evaluation**: Top-1 accuracy, confusion matrix, test loss

### Phase 2: Continuous Learning via Feedback Loop

* **Inputs**:

  * Real-time student drawings
  * Teacher corrections and new reference sketches
* **Mechanism**:

  * Logging corrected data
  * Periodic retraining using teacher-labeled samples
  * Triggered model replacement if new version performs better

### Phase 3: Deployment & MLOps

* **Deployment**: Streamlit Web App with modular backend
* **Modules**:

  * `preprocessing.py`: Image normalization & resizing
  * `model.py`: CNN-based sketch classifier
  * `training.py`: Model trainer and evaluator
  * `color_fill.py`: Flood fill for real-time color interactions
  * `sketch_completion.py`: Matching-based evaluation of drawing completions
  * `reinforcement.py`: Self-learning retraining from feedback
  * `api.py`: Unified API for frontend-backend communication
  * `app.py`: Streamlit application logic
* **MLOps**:

  * Model versioning, registry
  * Feedback pipeline
  * Auto-retraining triggers

---

## 🚸 Streamlit Deployment Guide

### Requirements

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

### Requirements File

```
streamlit
opencv-python
torch
torchvision
numpy
pillow
scikit-learn
pandas
matplotlib
streamlit-drawable-canvas
gdown
```

### Handling Large Datasets

* **Use Google Drive:**

  * Upload zipped Sketchy dataset to Drive
  * Use `gdown` to download it on runtime

---

## 🌟 Key Challenges & AI Considerations

* **Data Diversity**: Age group variations, artistic differences
* **Language Neutrality**: Uniform performance for English & Telugu words
* **Latency Constraints**: Real-time inference feedback
* **Offline Support**: Consider lightweight quantized models
* **Ethical AI**: Anonymization of student data, respectful handling of teacher feedback
* **Interpretability (Optional)**: Model explainability for better hint generation

---

## 🚀 Final Outcome

A real-time, sketch-based vocabulary learning tool that adapts to real student behavior, provides smart feedback, enables creative interaction, and evolves through continuous teacher guidance and feedback loops.

---

## 👨‍💼 Project Contributor

