import cv2
import numpy as np
import tensorflow as tf
from collections import deque
 
# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
MODEL_PATH = "fruit_ripeness_mobilenet.h5"
CONFIDENCE_THRESHOLD = 55.0
PREDICTION_INTERVAL = 5
SMOOTHING_WINDOW = 10
 
# -----------------------------
# CLASS NAMES (confirmed correct order from dataset)
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
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded | {len(class_names)} classes ready")
 
# -----------------------------
# SMOOTHING BUFFER
# -----------------------------
pred_buffer = deque(maxlen=SMOOTHING_WINDOW)
 
# -----------------------------
# OPEN CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()
 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Camera started. Press 'q' to quit.")
 
frame_count = 0
last_label     = "Waiting..."
last_confidence = 0.0
last_ripeness  = ""
last_color     = (200, 200, 200)
 
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame.")
        break
 
    frame = cv2.flip(frame, 1)          # mirror effect
    h, w  = frame.shape[:2]
    frame_count += 1
 
    # ----------------------------------------
    # CENTER ROI GUIDE BOX
    # ----------------------------------------
    roi_size = min(h, w) - 80
    cx, cy   = w // 2, h // 2
    x1 = cx - roi_size // 2
    y1 = cy - roi_size // 2
    x2 = cx + roi_size // 2
    y2 = cy + roi_size // 2
 
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(frame, "Place fruit here",
                (x1 + 10, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
 
    # ----------------------------------------
    # PREDICT EVERY N FRAMES
    # ----------------------------------------
    if frame_count % PREDICTION_INTERVAL == 0:
 
        roi = frame[y1:y2, x1:x2]
 
        # ✅ CRITICAL FIX: BGR -> RGB (MobileNet trained on RGB)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
 
        img = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
 
        preds = model.predict(img, verbose=0)[0]
        pred_buffer.append(preds)
 
        avg_preds  = np.mean(pred_buffer, axis=0)
        confidence = float(np.max(avg_preds)) * 100
        idx        = int(np.argmax(avg_preds))
 
        if confidence < CONFIDENCE_THRESHOLD:
            last_label      = "No fruit detected"
            last_ripeness   = "Show a fruit inside the yellow box"
            last_confidence = confidence
            last_color      = (0, 0, 220)           # red
        else:
            label      = class_names[idx]
            fruit      = label.split("_")[0].capitalize()
            ripeness   = label.split("_")[1].capitalize()
            last_label      = f"{fruit} - {ripeness}"
            last_ripeness   = ripeness_days.get(label, "Unknown")
            last_confidence = confidence
            # green = ripe, orange = unripe
            last_color = (0, 210, 0) if ripeness == "Ripe" else (0, 140, 255)
 
    # ----------------------------------------
    # DRAW RESULT PANEL (bottom bar)
    # ----------------------------------------
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 100), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
 
    # Fruit name + confidence
    cv2.putText(frame,
        f"{last_label}  ({last_confidence:.1f}%)",
        (20, h - 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, last_color, 2, cv2.LINE_AA)
 
    # Ripeness info
    cv2.putText(frame,
        last_ripeness,
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 80), 2, cv2.LINE_AA)
 
    cv2.imshow("Fruit Ripeness Detection", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
print("Camera closed.")