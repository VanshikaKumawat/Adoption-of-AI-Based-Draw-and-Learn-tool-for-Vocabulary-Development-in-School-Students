# api.py

import model
import training
import reinforcement
import preprocessing
import sketch_completion

# API: Preprocess an uploaded sketch
def preprocess_sketch(image):
    return preprocessing.preprocess_image(image)

# API: Predict class of sketch
def classify_sketch(image):
    processed = preprocess_sketch(image)
    prediction = model.predict(processed)
    return prediction

# API: Train model
def train_model(train_dir, test_dir, model_save_path='best_model.pth'):
    return training.train(train_dir, test_dir, model_save_path)

# API: Evaluate model
def evaluate_model(model_path, test_dir):
    return training.evaluate(model_path, test_dir)

# API: Provide feedback loop
def submit_teacher_feedback(image, true_label, pred_label):
    return reinforcement.log_teacher_feedback(image, true_label, pred_label)

# API: Trigger model update from feedback
def retrain_from_feedback(feedback_csv, model_save_path='student_model_updated.pth'):
    return reinforcement.retrain_from_feedback(feedback_csv, model_save_path)

# API: Sketch Completion placeholder (update later if needed)
def complete_drawing(sketch):
    return sketch_completion.complete_drawing_interactive(sketch)
