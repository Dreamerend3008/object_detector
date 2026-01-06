import os
# Force Qt to use xcb instead of wayland
os.environ["QT_QPA_PLATFORM"] = "xcb"

from ultralytics import YOLO
import cv2
from pathlib import Path

# Get the script's directory to build absitolute path
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR.parent / "models" / "objects_v1" / "weights" / "best.pt"

# Load model
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

print("Press 'q' to quit")

# Class names
CLASS_NAMES = ['chair', 'knife', 'phone']

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection (lower confidence to 0.25 to detect more)
    results = model(frame, conf=0.25, verbose=False)
    
    # Debug: print detections
    boxes = results[0].boxes
    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            print(f"Detected: {cls_name} ({conf:.2f})")
    
    # Draw results
    annotated = results[0].plot()
    
    # Add text showing detection count
    det_count = len(boxes)
    cv2.putText(annotated, f"Detections: {det_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display
    cv2.imshow('Webcam Detection', annotated)
    
    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()