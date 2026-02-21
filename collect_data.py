import cv2
import os
import time

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z', 'hello', 'thanks', 'yes', 'no']

IMAGES_PER_LABEL = 100

for label in labels:
    folder = os.path.join('data', 'images', label)
    os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)

print("\n=== Sign Language Data Collector ===")
print("Press SPACE to start collecting for each sign")
print("Press Q to quit\n")

for label in labels:
    print(f"Get ready to show sign: {label}")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'GET READY: {label}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, 'Press SPACE to start', (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('Data Collector', frame)

        key = cv2.waitKey(1)
        if key == ord(' '):
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    count = 0
    while count < IMAGES_PER_LABEL:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        img_path = os.path.join('data', 'images', label, f'{count}.jpg')
        cv2.imwrite(img_path, frame)
        count += 1

        cv2.putText(frame, f'Collecting: {label} [{count}/{IMAGES_PER_LABEL}]', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Data Collector', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    print(f"Done: {label}")
    time.sleep(1)

cap.release()
cv2.destroyAllWindows()
print("\nAll data collected!")