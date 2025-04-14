import cv2
from ultralytics import YOLO
import time

# Load YOLO11n (nano - fastest version)
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv11 inference (resize input for speed)
    results = model.predict(frame, imgsz=640, verbose=False)[0]
    
    # Draw boxes
    annotated = results.plot()

    # Show result
    cv2.imshow("YOLO11 Live Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
