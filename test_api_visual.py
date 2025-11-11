import requests
import cv2
import numpy as np

# YOLO Flask API endpoint
url = 'http://127.0.0.1:5000/detect'

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Encode frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(url, files={'image': img_encoded.tobytes()})

    try:
        detections = response.json()
    except:
        detections = []

    for det in detections:
        x1, y1, x2, y2 = det['box']

        # Scale coordinates if they are normalized (0-1)
        if max(x1, y1, x2, y2) <= 1.0:
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
        else:  # already in pixels
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show video with detections
    cv2.imshow("YOLO Detections", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
