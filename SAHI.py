# https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/
import numpy as np
import cv2
from ultralytics import YOLO
# import random

# # opening the file in read mode
my_file = open("coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = [(255, 0, 0), (255, 0, 255), (255, 255, 0), (255, 255, 255)]
# for i in range(len(class_list)):
#     r = random.randint(0, 255)
#     g = random.randint(0, 255)
#     b = random.randint(0, 255)
#     detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO("../models/yolov8_1.pt", "v8")

# Vals to resize video frames | small frame optimise the run
frame_wid = 640
frame_hyt = 640

cap = cv2.VideoCapture("../Video/Testing/Test1.mp4")
# cap = cv2.VideoCapture("Video tanpa judul ‐ Dibuat dengan Clipchamp.mp4")
# cap = cv2.VideoCapture("Hijau 103.png")
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #  resize the frame | small frame optimise the run
    frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    detect_params = model.predict(
        source=[frame], conf=0.25, save=False, imgsz=640,)
    # source=[frame], conf=0.4, save=False, imgsz=640, optimize=True)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    # print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            # print(i)
            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(
                bb[2]), int(bb[3])), detection_colors[int(clsID)], 3,)

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, class_list[int(clsID)] + " " + str(round(
                conf, 3)) + "%", (int(bb[0]), int(bb[1]) - 10), font, 0.5, detection_colors[int(clsID)], 2,)

    # Display the resulting frame
    cv2.imshow('ObjectDetection', frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()