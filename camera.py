import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import av

st.title("😎 Real-Time Face Filter (Sunglasses) - WebRTC")

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load sunglasses filter with alpha channel
filter_img = cv2.imread('sunglasses.png', cv2.IMREAD_UNCHANGED)

if filter_img is None:
    st.error("❌ Couldn't load sunglasses.png. Make sure the file exists in the same folder.")
    st.stop()

brightness = st.slider("Adjust Brightness", 0, 255, 128)

# Face filter logic
def overlay_filter(frame, filter_img, x, y, w, h):
    try:
        filter_resized = cv2.resize(filter_img, (w, int(h / 2)))
        fh, fw, _ = filter_resized.shape

        roi = frame[y:y+fh, x:x+fw]

        filter_rgb = filter_resized[:, :, :3]
        alpha_mask = filter_resized[:, :, 3] / 255.0

        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask) + filter_rgb[:, :, c] * alpha_mask

        frame[y:y+fh, x:x+fw] = roi
    except:
        pass

    return frame

# Video transformer class
class FaceFilterTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.convertScaleAbs(img, alpha=brightness / 128)  # Adjust brightness

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            offset_y = 25
            offset_x = 25
            img = overlay_filter(img, filter_img, x + offset_x, y + offset_y, w, h)

        return img

# Start webcam
webrtc_streamer(key="face-filter", video_transformer_factory=FaceFilterTransformer)
