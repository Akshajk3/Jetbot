from controller import Robot
import torch
from ultralytics import YOLO
import numpy as np
import cv2


# torch
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# yolo
model = YOLO("yolov8n.pt")
model.to(device)


# webots
TIME_STEP = 32

MAX_SPEED = 6.28

robot = Robot()

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
camera = robot.getDevice("camera")
camera.enable(64)

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.1 * MAX_SPEED)
right_motor.setVelocity(-0.1 * MAX_SPEED)


# update loop
while robot.step(TIME_STEP) != -1:
    img = camera.getImage()

    if img is None:
        continue

    width = camera.getWidth()
    height = camera.getHeight()

    frame = np.frombuffer(img, dtype=np.uint8).reshape((height, width, 4)).copy()

    frame = frame[:, :, :3]

    frame_resized = cv2.resize(frame, (320, 320))
    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    frame_tensor = frame_tensor.to(device)

    results = model(frame_tensor, verbose=False)

    detections = results[0].boxes

    for box in detections:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0]

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{label} {conf:.2f}"
        cv2.putText(frame_resized, text, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Webots YOLO Detection", frame_resized)
    cv2.waitKey(1)

cv2.destroyAllWindows()