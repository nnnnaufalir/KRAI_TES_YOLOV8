import numpy as np
import cv2
from ultralytics import YOLO
import time

# Load class list from file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Define detection colors
detection_colors = [(255, 0, 0), (255, 0, 255), (255, 255, 0), (255, 255, 255)]

# Load YOLOv8 model
model = YOLO("../models/yolov8_1.pt", "v8")

# Video frame dimensions
frame_width = 640
frame_height = 640

cap = cv2.VideoCapture("../Video/Testing/Test2.mp4")
if not cap.isOpened():
    print("Cannot open video file")
    exit()

max_detections = 10  # Maximum number of detections to display in a frame

frame_count = 0
start_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("End of video stream")
        break

    # Resize frame
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Predict on frame
    detect_params = model.predict(
        source=[frame], conf=0.25, save=False, imgsz=640)

    # Convert tensor array to numpy
    detections = detect_params[0].numpy()

    # List to store filtered detections
    filtered_detections = []

    if len(detections) != 0:
        for detection in detections:
            box = detection.boxes[0]
            clsID = box.cls[0]
            conf = box.conf[0]
            bb = box.xyxy[0]
            filtered_detections.append((bb, clsID, conf))

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            [d[0] for d in filtered_detections],  # Bounding boxes
            [d[2] for d in filtered_detections],  # Confidences
            score_threshold=0.5,
            nms_threshold=0.4
        )

        # Draw filtered detections
        for i in indices:
            bb, clsID, conf = filtered_detections[i]
            # Draw bounding box
            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                          detection_colors[int(clsID)], 3)

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, f"{class_list[int(clsID)]} {round(conf * 100, 2)}%",
                        (int(bb[0]), int(bb[1]) - 10), font, 0.6,
                        detection_colors[int(clsID)], 2)

    # Increment frame count
    frame_count += 1

    # Calculate FPS
    end_time = time.time()
    fps = frame_count / (end_time - start_time)

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display frame
    cv2.imshow('ObjectDetection', frame)

    # Terminate when "Q" is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
