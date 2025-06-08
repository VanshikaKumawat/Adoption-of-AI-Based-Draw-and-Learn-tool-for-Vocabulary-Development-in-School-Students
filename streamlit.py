# streamlit.py

import streamlit as st
from color_fill import color_fill_app
from sketch_completion import sketch_completion_app
from reinforcement import reinforcement_app
from utils import show_intro

# App Title
st.set_page_config(page_title="Draw-and-Learn AI", layout="wide")
st.title("🖍️ Draw-and-Learn AI Tool for Vocabulary Development")

# Sidebar Navigation
st.sidebar.title("Navigation")
choice = st.sidebar.radio(
    "Go to", ["Introduction", "Color Fill", "Sketch Completion", "Reinforcement Learning"]
)

# Load the selected section
if choice == "Introduction":
    show_intro()

elif choice == "Color Fill":
    color_fill_app()

elif choice == "Sketch Completion":
    sketch_completion_app()

elif choice == "Reinforcement Learning":
    reinforcement_app()
