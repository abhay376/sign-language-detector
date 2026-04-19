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
word_buffer = []  # Store detected letters to form words

# ===== PANEL DIMENSIONS =====
CAM_WIDTH = 480
CAM_HEIGHT = 360
MASK_SIZE = 200
PANEL_WIDTH = CAM_WIDTH + MASK_SIZE + 40  # total width
PANEL_HEIGHT = CAM_HEIGHT + 160  # extra space for bottom info

def create_hand_mask(frame, hand_landmarks, h, w):
    """Create a binary mask of the hand region."""
    # Get bounding box of hand
    x_coords = [int(lm.x * w) for lm in hand_landmarks]
    y_coords = [int(lm.y * h) for lm in hand_landmarks]
    
    pad = 30
    x_min = max(0, min(x_coords) - pad)
    x_max = min(w, max(x_coords) + pad)
    y_min = max(0, min(y_coords) - pad)
    y_max = min(h, max(y_coords) + pad)
    
    # Crop hand region
    hand_region = frame[y_min:y_max, x_min:x_max]
    
    if hand_region.size == 0:
        return np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
    
    # Convert to grayscale and apply threshold for skin detection
    hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
    # Skin color range in HSV
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Resize to display size
    mask = cv2.resize(mask, (MASK_SIZE, MASK_SIZE))
    return mask


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_timestamp += 33  # ~30fps
    landmarker.detect_async(mp_image, frame_timestamp)

    h, w, _ = frame.shape
    hand_mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)

    result = latest_result[0]
    if result and result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]

        # Draw landmarks manually
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
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 255), 2)
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

        # Draw bounding box around hand
        x_coords = [lm.x * w for lm in hand_landmarks]
        y_coords = [lm.y * h for lm in hand_landmarks]
        x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
        y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Create hand mask
        hand_mask = create_hand_mask(frame, hand_landmarks, h, w)

    # ===== BUILD THE MULTI-PANEL DISPLAY =====
    
    # Create main canvas (dark background)
    canvas = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 25)  # Dark background
    
    # --- LEFT: Main webcam feed ---
    canvas[10:10+CAM_HEIGHT, 10:10+CAM_WIDTH] = frame
    # Border around webcam
    cv2.rectangle(canvas, (9, 9), (11+CAM_WIDTH, 11+CAM_HEIGHT), (0, 255, 0), 1)
    
    # --- RIGHT TOP: Hand Mask ---
    mask_x = CAM_WIDTH + 25
    mask_y = 10
    # Convert mask to BGR for display
    mask_colored = cv2.cvtColor(hand_mask, cv2.COLOR_GRAY2BGR)
    canvas[mask_y:mask_y+MASK_SIZE, mask_x:mask_x+MASK_SIZE] = mask_colored
    cv2.rectangle(canvas, (mask_x-1, mask_y-1), (mask_x+MASK_SIZE+1, mask_y+MASK_SIZE+1), (100, 100, 100), 1)
    cv2.putText(canvas, "Hand Mask", (mask_x + 50, mask_y + MASK_SIZE + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # --- RIGHT BOTTOM: Prediction Display ---
    pred_y = mask_y + MASK_SIZE + 35
    if current_prediction:
        # Predicted text label
        cv2.putText(canvas, "Predicted:", (mask_x, pred_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        
        # Large prediction letter
        cv2.putText(canvas, str(current_prediction), (mask_x + 20, pred_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
        
        # Confidence bar
        bar_y = pred_y + 70
        bar_w = MASK_SIZE - 10
        bar_fill = int(bar_w * (confidence_score / 100))
        cv2.rectangle(canvas, (mask_x, bar_y), (mask_x + bar_w, bar_y + 12), (50, 50, 50), -1)
        
        # Color bar based on confidence
        if confidence_score > 80:
            bar_color = (0, 255, 0)
        elif confidence_score > 50:
            bar_color = (0, 200, 255)
        else:
            bar_color = (0, 0, 255)
        cv2.rectangle(canvas, (mask_x, bar_y), (mask_x + bar_fill, bar_y + 12), bar_color, -1)
        cv2.putText(canvas, f"{confidence_score:.1f}%", (mask_x + bar_w + 5, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    else:
        cv2.putText(canvas, "Show hand...", (mask_x, pred_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    # --- BOTTOM BAR: Word Buffer & Instructions ---
    bottom_y = CAM_HEIGHT + 25
    
    # Separator line
    cv2.line(canvas, (10, bottom_y), (PANEL_WIDTH - 10, bottom_y), (50, 50, 60), 1)
    
    # Word buffer display
    cv2.putText(canvas, "Detected Letters:", (15, bottom_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 170), 1)
    
    # Show last 15 stable predictions as a "word"
    if current_prediction and confidence_score > 60:
        if len(word_buffer) == 0 or word_buffer[-1] != current_prediction:
            word_buffer.append(current_prediction)
    if len(word_buffer) > 15:
        word_buffer = word_buffer[-15:]
    
    word_text = " ".join(word_buffer[-15:])
    cv2.putText(canvas, word_text, (15, bottom_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Instructions
    cv2.putText(canvas, "Q: Quit  |  C: Clear buffer  |  SPACE: Add space", 
                (15, bottom_y + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 120), 1)

    # FPS display
    cv2.putText(canvas, f"MediaPipe + RandomForest", (PANEL_WIDTH - 220, bottom_y + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 100), 1)

    cv2.imshow('Sign Language Detector', canvas)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        word_buffer = []
    elif key == ord(' '):
        word_buffer.append(' ')

cap.release()
landmarker.close()
cv2.destroyAllWindows()
print("Detector closed.")