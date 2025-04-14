import cv2
import numpy as np

# Load pre-trained model and config (MobileNet SSD)
prototxt = 'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/deploy.prototxt'
model = 'https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel'

# Download model files if not present
import os
import urllib.request

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

download_file(prototxt, 'deploy.prototxt')
download_file(model, 'MobileNetSSD_deploy.caffemodel')

# Load class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load the model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

# Start video stream
cap = cv2.VideoCapture(0)

print("[INFO] Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Preprocess input image
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, 
                                 (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Draw results on the frame
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            label = f"{CLASSES[idx]}: {confidence*100:.1f}%"
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Show frame
    cv2.imshow("Camera View", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
