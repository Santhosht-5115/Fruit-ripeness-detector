import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# 1️⃣ Load Trained MobileNetV2 Model
# ==============================
model = tf.keras.models.load_model("fruit_ripeness_mobilenet.h5")

# ==============================
# 2️⃣ Create Validation Generator
# ==============================
IMG_SIZE = 224
BATCH_SIZE = 32

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    "dataset",              # change if your validation folder is different
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False           # IMPORTANT: keep False for correct metrics
)

# ==============================
# 3️⃣ Get True Labels
# ==============================
y_true = val_generator.classes

# ==============================
# 4️⃣ Get Predictions
# ==============================
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# ==============================
# 5️⃣ Confusion Matrix
# ==============================
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n")
print(cm)

# ==============================
# 6️⃣ Classification Report
# ==============================
class_labels = list(val_generator.class_indices.keys())

report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels
)

print("\nClassification Report:\n")
print(report)