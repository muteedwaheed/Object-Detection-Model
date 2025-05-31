import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# Load the model
model = YOLO("best.pt")
model.conf = 0.5

st.markdown("<h1 style='text-align: center; color: black;'>🎯 Object Detection Model</h1>", unsafe_allow_html=True)

option = st.radio("Choose input method", ["📁 Upload Image", "📷 Live Camera"])

def detect_and_display(image_np, original_image):
    results = model(image_np)
    image_with_boxes = image_np.copy()
    for result in results:
        boxes = result.boxes
        names = result.names
        for box in boxes:
            conf = box.conf[0].item()
            if conf >= 0.5:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                label = f"{names[cls_id]} {conf:.2f}"
                cv2.rectangle(image_with_boxes, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                cv2.putText(image_with_boxes, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    st.markdown("### 🖼 Detection Results")
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(image_with_boxes, caption="Detection Result", use_container_width=True)

# Image Upload
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        detect_and_display(image_np, image)

# Live Camera
elif option == "📷 Live Camera":
    cam_placeholder = st.empty()  # Placeholder to hold camera input
    camera_image = cam_placeholder.camera_input("Take a picture")
    if camera_image is not None:
        cam_placeholder.empty()  # Clear camera preview after capture
        image = Image.open(camera_image).convert("RGB")
        image_np = np.array(image)
        detect_and_display(image_np,image)