import cv2
import numpy as np
import pickle
import time
import mediapipe as mp

with open('model/sign_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']

# Setup mediapipe HandLandmarker (new Tasks API) for VIDEO mode
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# We use LIVE_STREAM mode for real-time detection
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

print("=== Sign Language Detector Running ===")
print("Press 'Q' to quit\n")

prediction_history = []
HISTORY_LENGTH = 10
current_prediction = ""
confidence_score = 0
frame_timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_timestamp += 33  # ~30fps
    landmarker.detect_async(mp_image, frame_timestamp)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)

    result = latest_result[0]
    if result and result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]

        # Draw landmarks manually
        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        # Draw connections
        HAND_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (255, 255, 255), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 255, 0), -1)

        # Extract landmarks for prediction
        landmarks = []
        for lm in hand_landmarks:
            landmarks.append(lm.x)
            landmarks.append(lm.y)

        landmarks = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(landmarks)[0]
        probabilities = model.predict_proba(landmarks)[0]
        confidence = max(probabilities) * 100

        prediction_history.append(prediction)
        if len(prediction_history) > HISTORY_LENGTH:
            prediction_history.pop(0)

        current_prediction = max(set(prediction_history), key=prediction_history.count)
        confidence_score = confidence

        x_coords = [lm.x * w for lm in hand_landmarks]
        y_coords = [lm.y * h for lm in hand_landmarks]
        x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
        y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    if current_prediction:
        cv2.putText(frame, f'Sign: {current_prediction}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f'Confidence: {confidence_score:.1f}%', (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(frame, 'Show your hand...', (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    cv2.putText(frame, 'Press Q to quit', (frame.shape[1] - 200, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    cv2.imshow('Sign Language Detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
print("Detector closed.")