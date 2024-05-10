import os
import cv2
from ultralytics import YOLO


model_path = '../models/yolov8_0.pt'

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.1

class_name_dict = {0: 'bolaBiru',
                   1: 'bolaUngu',
                   2: 'padiBiru',
                   3: 'silo'}

cap = cv2.VideoCapture("../Video/Testing/Test1.mp4")  # use default camera
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Real-time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
