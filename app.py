import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
 
# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Fruit Ripeness Detection",
    page_icon="🍎",
    layout="wide"
)
 
# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
MODEL_PATH = "fruit_ripeness_mobilenet.h5"
 
# -----------------------------
# CLASS NAMES (confirmed correct order)
# -----------------------------
class_names = [
    'apple_ripe', 'apple_unripe',
    'banana_ripe', 'banana_unripe',
    'grapes_ripe', 'grapes_unripe',
    'guava_ripe', 'guava_unripe',
    'mango_ripe', 'mango_unripe',
    'orange_ripe', 'orange_unripe',
    'papaya_ripe', 'papaya_unripe',
    'pineapple_ripe', 'pineapple_unripe',
    'tomato_ripe', 'tomato_unripe',
    'watermelon_ripe', 'watermelon_unripe'
]
 
# -----------------------------
# RIPENESS INFO
# -----------------------------
ripeness_days = {
    "apple_unripe":      "4-6 days to ripen",
    "banana_unripe":     "2-3 days to ripen",
    "grapes_unripe":     "3-5 days to ripen",
    "guava_unripe":      "2-4 days to ripen",
    "mango_unripe":      "3-6 days to ripen",
    "orange_unripe":     "5-7 days to ripen",
    "papaya_unripe":     "2-3 days to ripen",
    "pineapple_unripe":  "4-6 days to ripen",
    "tomato_unripe":     "2-4 days to ripen",
    "watermelon_unripe": "5-7 days to ripen",
    "apple_ripe":        "Ready to eat!",
    "banana_ripe":       "Ready to eat!",
    "grapes_ripe":       "Ready to eat!",
    "guava_ripe":        "Ready to eat!",
    "mango_ripe":        "Ready to eat!",
    "orange_ripe":       "Ready to eat!",
    "papaya_ripe":       "Ready to eat!",
    "pineapple_ripe":    "Ready to eat!",
    "tomato_ripe":       "Ready to eat!",
    "watermelon_ripe":   "Ready to eat!",
}
 
# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)
 
model = load_model()
 
# -----------------------------
# HELPERS
# -----------------------------
def preprocess(image: Image.Image):
    """PIL Image (RGB) -> model input."""
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)
 
def predict(img_array, threshold):
    preds      = model.predict(img_array, verbose=0)
    confidence = float(np.max(preds)) * 100
    idx        = int(np.argmax(preds))
    label      = class_names[idx]
    return {
        "detected":   confidence >= threshold,
        "fruit":      label.split("_")[0].capitalize(),
        "ripeness":   label.split("_")[1].capitalize(),
        "confidence": confidence,
        "info":       ripeness_days.get(label, "N/A")
    }
 
def show_result(result):
    if not result["detected"]:
        st.error(f"No fruit detected — confidence {result['confidence']:.1f}% is below threshold.")
        st.info("Try lowering the Confidence Threshold in the sidebar or improve lighting.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.success(f"🍎 Fruit: **{result['fruit']}**")
    with c2:
        icon = "🟢" if result["ripeness"] == "Ripe" else "🟠"
        st.info(f"{icon} Ripeness: **{result['ripeness']}**")
    st.progress(int(result["confidence"]))
    st.write(f"**Confidence:** {result['confidence']:.2f}%")
    st.warning(f"**Ripeness Info:** {result['info']}")
 
# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Settings")
CONFIDENCE_THRESHOLD = st.sidebar.slider(
    "Confidence Threshold (%)", 0, 100, 55
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Model:** MobileNetV2\n\n"
    "**Input Size:** 224×224\n\n"
    "**Classes:** 20 (10 fruits × ripe/unripe)"
)
 
# -----------------------------
# TITLE
# -----------------------------
st.markdown(
    "<h1 style='text-align:center;'>🍎 Fruit Ripeness Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Detect fruit type and ripeness using Deep Learning</p>",
    unsafe_allow_html=True
)
st.markdown("---")
 
# -----------------------------
# INPUT METHOD
# -----------------------------
option = st.radio(
    "Select Input Method:",
    ["Upload Image", "Camera Snapshot", "Live Camera (OpenCV)"],
    horizontal=True
)
st.markdown("---")
 
# ==========================================================
# 1. UPLOAD IMAGE
# ==========================================================
if option == "Upload Image":
    uploaded = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            with st.spinner("Analyzing..."):
                result = predict(preprocess(image), CONFIDENCE_THRESHOLD)
            show_result(result)
 
# ==========================================================
# 2. CAMERA SNAPSHOT
#    Streamlit's built-in camera_input — stable, no flickering
# ==========================================================
elif option == "Camera Snapshot":
    st.info("📸 Click **Take Photo** — then the result will appear instantly.")
    snapshot = st.camera_input("Take a photo of the fruit")
    if snapshot:
        image = Image.open(snapshot).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Captured Image", use_container_width=True)
        with col2:
            with st.spinner("Analyzing..."):
                result = predict(preprocess(image), CONFIDENCE_THRESHOLD)
            show_result(result)
 
# ==========================================================
# 3. LIVE CAMERA — launches camera_predict.py in OpenCV window
#    This is the CORRECT way to do smooth live camera.
#    Streamlit is not designed for real-time video streams.
# ==========================================================
elif option == "Live Camera (OpenCV)":
    st.markdown("### 🎥 Live Real-Time Camera")
    st.info(
        "Streamlit cannot run smooth live video inside the browser.\n\n"
        "Click the button below to launch the **Live Camera window** "
        "which runs directly on your PC with full real-time detection."
    )
 
    if st.button("▶ Launch Live Camera", use_container_width=True):
        import subprocess
        import sys
        subprocess.Popen([sys.executable, "camera_predict.py"])
        st.success(
            "✅ Live camera launched! "
            "A new OpenCV window should open on your screen.\n\n"
            "Press **Q** in that window to close it."
        )
 
    st.markdown("---")
    st.markdown("**How the Live Camera works:**")
    st.markdown(
        "- Hold your fruit inside the **yellow box** on screen\n"
        "- The model predicts every 5 frames for stability\n"
        "- Result shows at the bottom: Fruit name, Ripeness, Confidence %\n"
        "- Press **Q** to quit"
    )