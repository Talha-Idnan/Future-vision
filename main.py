
import cv2
from ultralytics import YOLO
from facenet_pytorch import MTCNN
import torch

import time

# Device and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = MTCNN(keep_all=True, device=device)
model = YOLO("yolo26n.pt")

# Capture webcam frames
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Could not read camera frame")
        break

    start_time = time.perf_counter()

    # Face detection using MTCNN
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probabilities = model1.detect(rgb_frame)
    

    if boxes is not None and probabilities is not None:
        for box, probability in zip(boxes, probabilities):
            if probability is None or probability < 0.90:
                continue

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"Face: {probability:.2f}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Object detection using YOLO
    results = model(frame, verbose=False)

    # Plot YOLO detections on the same frame
    # result 0 means sesult for webcam 1

    
    annotated_frame = results[0].plot()
    elapsed_time = time.perf_counter() - start_time
    fps = 1 / elapsed_time
    cv2.putText(
    annotated_frame,
    f"FPS: {fps:.1f}",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)


#  for fps 

    cv2.putText(
    annotated_frame,
    f"FPS: {fps:.1f}",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

    # Display the final frame
    cv2.imshow("Future Vision - Object and Face Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources after the loop ends
cap.release()
cv2.destroyAllWindows()
