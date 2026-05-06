from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ✅ Create list to store detections
detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform object detection
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            label = model.names[class_id]

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # ✅ Save detection data
            detections.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Object": label,
                "Confidence": round(confidence, 2)
            })

    cv2.imshow("ObjectScope - Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ✅ Save CSV after loop ends
if detections:
    df = pd.DataFrame(detections)
    df.to_csv("detections.csv", index=False)
    print("✅ detections.csv saved successfully")
else:
    print("❌ No detections found")