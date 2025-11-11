def run_yolo_detection(img):
    # img: OpenCV image
    # Perform YOLO detection here
    detections = [
        {"class": "person", "confidence": 0.98, "box": [100, 50, 200, 400]},
        {"class": "mask", "confidence": 0.95, "box": [120, 60, 180, 200]}
    ]
    return detections
