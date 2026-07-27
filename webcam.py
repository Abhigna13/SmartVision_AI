from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Draw detection results
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow("SmartVision AI - Webcam Detection", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release camera
cap.release()

# Close windows
cv2.destroyAllWindows()