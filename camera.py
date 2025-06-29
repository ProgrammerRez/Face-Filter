import streamlit as st
import cv2
import numpy as np

st.title("😎 Real-Time Face Filter (Sunglasses)")

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load sunglasses filter (with alpha channel)
filter_img = cv2.imread('sunglasses.png', cv2.IMREAD_UNCHANGED)

if filter_img is None:
    st.error("Couldn't load sunglasses.png. Make sure the file exists in the same folder.")
    st.stop()

brightness = st.slider("Adjust Brightness", 0, 255, 128)
run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

def overlay_filter(frame, filter_img, x, y, w, h):
    try:
        # Resize filter to fit face width and partial height
        filter_resized = cv2.resize(filter_img, (w, int(h / 2)))
        fh, fw, _ = filter_resized.shape

        # Define region of interest
        roi = frame[y:y+fh, x:x+fw]

        # Separate color and alpha channels
        filter_rgb = filter_resized[:, :, :3]
        alpha_mask = filter_resized[:, :, 3] / 255.0

        # Blend using the alpha mask
        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask) + filter_rgb[:, :, c] * alpha_mask

        # Replace the ROI in the frame
        frame[y:y+fh, x:x+fw] = roi
    except:
        pass  # Prevent crash if filter goes out of bounds

    return frame

if run:
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_BRIGHTNESS, brightness / 255)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        camera.set(cv2.CAP_PROP_BRIGHTNESS, brightness / 255)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            if y > 0 and x > 0:
                offset_y = 25  # adjust this value to shift more/less
                offset_x = 25
                frame = overlay_filter(frame, filter_img, x+offset_x, y + offset_y, w, h)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(rgb_frame)

    camera.release()
else:
    st.warning("Check the box to start the webcam.")
