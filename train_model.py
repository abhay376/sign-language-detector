import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import mediapipe as mp

print("=== Sign Language Model Trainer ===\n")

# Setup mediapipe HandLandmarker (new Tasks API)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

data = []
labels = []

data_dir = os.path.join('data', 'images')

# Loop through each label folder
for label in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, label)
    if not os.path.isdir(folder_path):
        continue
    print(f"Processing: {label}")

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            data.append(landmarks)
            labels.append(label)

landmarker.close()

print(f"\nTotal samples collected: {len(data)}")

if len(data) == 0:
    print("\nNo hand landmarks detected in any images. Cannot train model.")
    exit(1)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Check if we have enough classes
unique_labels = np.unique(labels)
if len(unique_labels) < 2:
    print(f"\nOnly found {len(unique_labels)} class(es). Need at least 2 to train.")
    exit(1)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# Train Random Forest model
print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Save the model
os.makedirs('model', exist_ok=True)
model_path = os.path.join('model', 'sign_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"Model saved to {model_path}")
print("\nYou can now run app.py for real-time detection!")
