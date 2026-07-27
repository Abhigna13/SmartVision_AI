from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolov8n.pt")

# Detect objects in the image
results = model("images/street.jpg", save=True)

print("Object Detection Completed Successfully!")