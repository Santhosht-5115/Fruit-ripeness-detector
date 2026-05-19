import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Dataset path
DATASET_PATH = "dataset"
IMG_SIZE = 224

X = []
y = []

print("Loading images...")

# Loop through class folders (apple_ripe, banana_unripe, etc.)
for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)

    # Check if it is a directory
    if not os.path.isdir(class_path):
        continue

    print(f"Processing class: {class_name}")

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # Ensure it is a file
        if not os.path.isfile(img_path):
            continue

        try:
            img = cv2.imread(img_path)

            # Skip corrupted images
            if img is None:
                print("Invalid image:", img_path)
                continue

            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Normalize (0–1)
            img = img / 255.0

            X.append(img)
            y.append(class_name)

        except Exception as e:
            print("Error loading image:", img_path)
            print(e)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print("\nTotal images loaded:", len(X))

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Classes found:", label_encoder.classes_)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Save preprocessed data (optional but recommended)
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("\nPreprocessing completed successfully ✅")
