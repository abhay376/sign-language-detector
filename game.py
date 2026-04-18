import cv2
import numpy as np
import pickle
import time
import mediapipe as mp
import random

# Load the model
with open('model/sign_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']

# Extract available signs from the model's trained classes
available_signs = list(model.classes_)

# Setup MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = [None]
def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    latest_result[0] = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback
)

landmarker = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

# Game State Variables
score = 0
target_sign = random.choice(available_signs)
correct_time = 0
SHOW_CORRECT_PULSE = 1.5  # Time to show "Correct!" celebration
prediction_history = []
HISTORY_LENGTH = 10
frame_timestamp = 0

print("=== Sign Language Learning Game Started ===")
print("Press 'Q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_timestamp += 33
    landmarker.detect_async(mp_image, frame_timestamp)

    # Draw top header bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 150), (30, 30, 30), -1)

    current_time = time.time()
    
    # UI Logic: Check if we are currently celebrating a correct answer
    is_celebrating = (current_time - correct_time) < SHOW_CORRECT_PULSE
    
    if is_celebrating:
        cv2.putText(frame, 'CORRECT! +1', (frame.shape[1]//2 - 150, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 5)
    else:
        # Prompt user to perform the next sign
        cv2.putText(frame, f'Make this sign: {target_sign}', (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, f'Score: {score}', (frame.shape[1] - 250, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    result = latest_result[0]
    if result and result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]

        # Draw hand skeleton connecting lines
        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        HAND_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                            (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                            (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 255, 0), -1)

        # Predict current sign
        landmarks = []
        for lm in hand_landmarks:
            landmarks.append(lm.x)
            landmarks.append(lm.y)

        landmarks = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(landmarks)[0]
        probabilities = model.predict_proba(landmarks)[0]
        confidence = max(probabilities) * 100

        # Stabilize prediction over HISTORY_LENGTH frames
        prediction_history.append(prediction)
        if len(prediction_history) > HISTORY_LENGTH:
            prediction_history.pop(0)

        current_prediction = max(set(prediction_history), key=prediction_history.count)

        # Highlight current reading below the main prompt
        if not is_celebrating:
            if confidence > 65:
                cv2.putText(frame, f'AI sees: {current_prediction} ({confidence:.0f}%)', (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            else:
                cv2.putText(frame, 'AI sees: Unknown Sign', (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

            # Check if user correctly performed the target sign!
            if current_prediction == target_sign and confidence >= 75:
                score += 1
                correct_time = current_time
                old_sign = target_sign
                while target_sign == old_sign: # Make sure we get a new random sign Next
                    target_sign = random.choice(available_signs)
                prediction_history.clear() # Clear history to avoid fast-skipping

    cv2.putText(frame, 'Press Q to quit', (frame.shape[1] - 200, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    cv2.imshow('Sign Language Game', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
print("Game closed. Final Score:", score)
