import cv2
import os
import time

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z', 'hello', 'thanks', 'yes', 'no']

IMAGES_PER_LABEL = 200  # More samples for better accuracy
CAPTURE_DELAY_MS = 150  # Slow down capture so you can vary your hand position

for label in labels:
    folder = os.path.join('data', 'images', label)
    os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)

# Wait for camera to initialize
import time as _t
for _ in range(10):
    ret, test_frame = cap.read()
    if ret and test_frame is not None:
        break
    _t.sleep(0.5)
else:
    print("ERROR: Could not access webcam. Close any other apps using the camera and try again.")
    exit(1)

print("\n=== Sign Language Data Collector (Improved) ===")
print("TIPS FOR BETTER ACCURACY:")
print("  - MOVE your hand around the frame while collecting")
print("  - TILT your hand slightly at different angles")
print("  - CHANGE the distance (closer and farther)")
print("  - Use DIFFERENT backgrounds if possible")
print("\nPress SPACE to start collecting for each sign")
print("Press Q to quit\n")

for label in labels:
    # Check if we already have enough images for this label
    folder = os.path.join('data', 'images', label)
    existing = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
    if existing >= IMAGES_PER_LABEL:
        print(f"Skipping {label} - already has {existing} images")
        continue

    print(f"\nGet ready to show sign: {label}")
    print(f"  (Already have {existing} images, need {IMAGES_PER_LABEL - existing} more)")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Instruction overlay
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 130), (0, 0, 0), -1)
        cv2.putText(frame, f'GET READY: {label}', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, 'Press SPACE to start', (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, 'MOVE your hand around while collecting!', (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow('Data Collector', frame)

        key = cv2.waitKey(1)
        if key == ord(' '):
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    count = existing  # Start from where we left off
    while count < IMAGES_PER_LABEL:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        img_path = os.path.join('data', 'images', label, f'{count}.jpg')
        cv2.imwrite(img_path, frame)
        count += 1

        # Show progress with reminder to move hand
        progress = count / IMAGES_PER_LABEL
        bar_width = int(400 * progress)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
        cv2.putText(frame, f'Collecting: {label} [{count}/{IMAGES_PER_LABEL}]', (30, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, 'KEEP MOVING YOUR HAND!', (30, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        # Progress bar
        cv2.rectangle(frame, (30, 80), (30 + 400, 95), (50, 50, 50), -1)
        cv2.rectangle(frame, (30, 80), (30 + bar_width, 95), (0, 255, 0), -1)
        
        cv2.imshow('Data Collector', frame)

        # IMPORTANT: This delay gives you time to move your hand between captures
        if cv2.waitKey(CAPTURE_DELAY_MS) == ord('q'):
            break

    print(f"Done: {label} ({count} images)")
    time.sleep(1)

cap.release()
cv2.destroyAllWindows()
print("\nAll data collected! Now run: python train_model.py")