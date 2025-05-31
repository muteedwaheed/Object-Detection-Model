import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# Load the model
model = YOLO("best.pt")  # make sure this file is in the same folder

# Title
st.title("Dog Breed Detector 🐶")
st.markdown("Detects 6 dog breeds using your trained YOLOv8 model")

# Option selection
option = st.radio("Choose input method", ["📷 Live Camera", "📁 Upload Image"])

# Function to show detection results
def detect_and_display(image_np):
    results = model(image_np)
    for result in results:
        annotated = result.plot()  # returns an annotated image (numpy array)
        st.image(annotated, caption="Detection Result", use_column_width=True)

# Upload option
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        detect_and_display(image_np)

# Live camera option
elif option == "📷 Live Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Captured Image", use_column_width=True)
        detect_and_display(image_np)
