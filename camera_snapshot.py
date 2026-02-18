import cv2
import os
from ultralytics import YOLO
from datetime import datetime

# Load your trained model
model = YOLO("runs/detect/retrain_v2/weights/best.pt")

# Create folder for snapshots
os.makedirs("snapshots", exist_ok=True)

cap = cv2.VideoCapture(0)

snapshot_taken = False  # prevent multiple saves instantly

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    boxes = results[0].boxes

    # If at least one detection found
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            confidence = float(box.conf[0])

            if confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"ID Card {confidence:.2f}", 
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

                # Save snapshot once
                if not snapshot_taken:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"snapshots/id_detected_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Snapshot saved: {filename}")
                    snapshot_taken = True

    else:
        snapshot_taken = False  # reset if no ID visible

    cv2.imshow("ID Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
