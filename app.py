from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model (pretrained)
model = YOLO("yolov8s.pt")  # use your trained model if you have one

@app.route('/detect', methods=['POST'])
def detect():
    # Read image from request
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR

    results = model(img)
    detections = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()   # x1,y1,x2,y2
        scores = result.boxes.conf.cpu().numpy()  # confidence
        classes = result.boxes.cls.cpu().numpy()  # class indices

        height, width = img.shape[:2]

        for box, conf, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            label = model.names[int(cls)]
            detections.append({
                "class": label,
                "confidence": float(conf),
                "box": [x1/width, y1/height, x2/width, y2/height]  # normalized
            })

    return jsonify(detections)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
