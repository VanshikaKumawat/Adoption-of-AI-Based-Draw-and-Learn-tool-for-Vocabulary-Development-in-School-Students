import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="Draw and Learn AI", layout="wide")

st.title("Draw and Learn AI: Color-Fill, Sketch Completion & Feedback")

# --- Tabs for clean separation of functionalities ---
tab1, tab2, tab3 = st.tabs(["Color Fill", "Sketch Completion", "Teacher Feedback"])

# ---------- TAB 1: Color Fill ----------
with tab1:
    st.header("Color Fill Tool")

    color_fill_canvas = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Orange fill with transparency
        stroke_width=3,
        stroke_color="#000000",
        background_color="#ffffff",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="color_fill_canvas",
    )

    if st.button("Apply Color Fill"):
        if color_fill_canvas.image_data is not None:
            img = np.array(color_fill_canvas.image_data.convert("RGB"))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Basic color fill logic: Close gaps and fill connected areas
            thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]
            kernel = np.ones((5, 5), np.uint8)
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Mask original image with closed areas for fill simulation
            filled_img = cv2.bitwise_and(img, img, mask=closed)

            st.image(filled_img, caption="Color Fill Result", use_column_width=True)
        else:
            st.warning("Draw something on the canvas to apply color fill.")

# ---------- TAB 2: Sketch Completion ----------
with tab2:
    st.header("Sketch Completion")

    completion_choice = st.radio("Select input method:", ["Draw Partial Sketch", "Upload Partial Sketch"])

    partial_sketch_img = None

    if completion_choice == "Draw Partial Sketch":
        sketch_canvas = st_canvas(
            fill_color="rgba(0, 0, 255, 0.3)",  # Blue fill transparency
            stroke_width=3,
            stroke_color="#000000",
            background_color="#ffffff",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key="partial_sketch_canvas",
        )
        if sketch_canvas.image_data is not None:
            partial_sketch_img = sketch_canvas.image_data

    else:  # Upload
        uploaded_file = st.file_uploader("Upload a partial sketch image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            partial_sketch_img = Image.open(uploaded_file)
            st.image(partial_sketch_img, caption="Uploaded Partial Sketch", use_column_width=True)

    if st.button("Complete the Drawing"):
        if partial_sketch_img is not None:
            # Placeholder: just echo input for now
            st.image(partial_sketch_img, caption="Completed Drawing (Dummy)", use_column_width=True)
            st.info("Completion AI logic will be implemented here.")
        else:
            st.warning("Please provide a partial sketch (draw or upload).")

# ---------- TAB 3: Teacher Feedback ----------
with tab3:
    st.header("Teacher Feedback (Reinforcement Learning Data)")

    st.write("Provide feedback on AI predictions to improve the model.")

    student_drawn_img = st.file_uploader("Upload student drawing for feedback", type=["png", "jpg", "jpeg"])
    ai_prediction = st.text_input("AI Prediction", "")
    correct_label = st.text_input("Correct Label (if prediction is wrong)", "")
    feedback = st.radio("Is the AI prediction correct?", ["Yes", "No"])

    if st.button("Submit Feedback"):
        if student_drawn_img is None:
            st.warning("Upload the student drawing image.")
        elif ai_prediction == "":
            st.warning("Enter AI prediction.")
        elif feedback == "No" and correct_label.strip() == "":
            st.warning("Enter the correct label for wrong prediction.")
        else:
            # TODO: Save this feedback data in your data pipeline / backend storage
            st.success("Feedback submitted. Thank you!")

            # Example: save uploaded file and feedback metadata locally (or push to DB/cloud)
            # with open(f"feedback_logs/{student_drawn_img.name}", "wb") as f:
            #     f.write(student_drawn_img.getbuffer())

            # Save metadata logic goes here

# --- Sidebar with model info / loading ---
st.sidebar.title("Model & System Status")
st.sidebar.info("Models and pipelines will be integrated here.")
