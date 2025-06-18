import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# Load the model
model = YOLO("best.pt")
model.conf = 0.5

# Set page config
st.set_page_config(page_title="Object Detection App", layout="wide")

# Custom CSS for background, navbar, and detection result text
st.markdown("""
    <style>
     .stApp {
         background-image: url('https://www.shutterstock.com/image-vector/abstract-3d-camera-scanning-faces-260nw-1740272483.jpg');
         background-size: cover;
         background-repeat: no-repeat;
         background-attachment: fixed;
     }
     .navbar {
         background-color: #333;
         overflow: hidden;
         padding: 10px 20px;
         display: flex;
         justify-content: space-between;
         align-items: center;
     }
     .navbar-left {
         color: #f2f2f2;
         font-size: 20px;
     }
     .navbar-right {
         color: #f2f2f2;
         font-size: 20px;
     }
     .box {
         background-color: rgba(255, 255, 255, 0.85);
         padding: 30px;
         border-radius: 10px;
     }
     h3 {
         color: white !important;
     }
    </style>
    <div class="navbar">
        <div class="navbar-left">Home</div>
        <div class="navbar-right">🎯 Object Detection Model</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar for input method
st.sidebar.title("Select Input Method")
option = st.sidebar.radio("Options", ["📁 Upload Image", "📷 Live Camera"])

# Detection and display logic
def detect_and_display(image_np, original_image):
    results = model(image_np)
    image_with_boxes = image_np.copy()

    for result in results:
        boxes = result.boxes
        names = result.names
        if boxes is not None and len(boxes) > 0:
            max_conf_idx = boxes.conf.argmax()
            box = boxes[max_conf_idx]
            conf = box.conf[0].item()
            if conf >= 0.5:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)

                # Shrink the box to focus only on the face area
                box_width = xyxy[2] - xyxy[0]
                box_height = xyxy[3] - xyxy[1]
                xyxy[0] += int(box_width * 0.25)
                xyxy[2] -= int(box_width * 0.25)
                xyxy[1] += int(box_height * 0.25)
                xyxy[3] -= int(box_height * 0.1)

                cls_id = int(box.cls[0])
                label = f"{names[cls_id]} {conf:.2f}"
                cv2.rectangle(image_with_boxes, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                cv2.putText(image_with_boxes, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    st.markdown("### 🖼 Detection Results")
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Image", width=400)
    with col2:
        st.image(image_with_boxes, caption="Detection Result", width=400)

# Image Upload
if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        detect_and_display(image_np, image)

# Live Camera
elif option == "📷 Live Camera":
    cam_placeholder = st.empty()
    camera_image = cam_placeholder.camera_input("Take a picture")
    if camera_image is not None:
        cam_placeholder.empty()
        image = Image.open(camera_image).convert("RGB")
        image_np = np.array(image)
        detect_and_display(image_np, image)

# Close box wrapper
st.markdown('</div>', unsafe_allow_html=True)
