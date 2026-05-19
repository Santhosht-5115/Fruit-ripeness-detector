import tensorflow as tf
import numpy as np
import cv2
import os

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
MODEL_PATH = "fruit_ripeness_mobilenet.h5"
IMAGE_PATH = "test.jpg"

# -----------------------------
# RIPENESS DAYS LOGIC (STEP 3.1)
# -----------------------------
ripeness_days = {
    "apple_unripe": "4–6 days",
    "banana_unripe": "2–3 days",
    "grapes_unripe": "3–5 days",
    "guava_unripe": "2–4 days",
    "mango_unripe": "3–6 days",
    "orange_unripe": "5–7 days",
    "papaya_unripe": "2–3 days",
    "pineapple_unripe": "4–6 days",
    "tomato_unripe": "2–4 days",
    "watermelon_unripe": "5–7 days",

    "apple_ripe": "Ready to eat",
    "banana_ripe": "Ready to eat",
    "grapes_ripe": "Ready to eat",
    "guava_ripe": "Ready to eat",
    "mango_ripe": "Ready to eat",
    "orange_ripe": "Ready to eat",
    "papaya_ripe": "Ready to eat",
    "pineapple_ripe": "Ready to eat",
    "tomato_ripe": "Ready to eat",
    "watermelon_ripe": "Ready to eat"
}

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# Class names from dataset folder
class_names = sorted(os.listdir("dataset"))
print("✅ Class order:", class_names)

# -----------------------------
# LOAD & PREPROCESS IMAGE
# -----------------------------
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# -----------------------------
# PREDICTION
# -----------------------------
pred = model.predict(img)
pred_index = np.argmax(pred)
confidence = np.max(pred) * 100

predicted_class = class_names[pred_index]

# Ripeness-days result
ripening_info = ripeness_days.get(predicted_class, "N/A")

# -----------------------------
# OUTPUT
# -----------------------------
print(f"✅ Prediction: {predicted_class}")
print(f"🎯 Confidence: {confidence:.2f}%")
print(f"⏳ Ripeness Info: {ripening_info}")
