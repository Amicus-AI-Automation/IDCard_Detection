import cv2
from ultralytics import YOLO

# Load models
person_model = YOLO("yolov8n.pt")  # detects persons
id_model = YOLO("runs/detect/retrain_v2/weights/best.pt")  # your trained ID model

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run both models
    person_results = person_model(frame, conf=0.5)
    id_results = id_model(frame, conf=0.5)

    persons = person_results[0].boxes
    id_cards = id_results[0].boxes

    # Draw person boxes
    for p in persons:
        cls = int(p.cls[0])
        
        # COCO class 0 = person
        if cls == 0:
            x1, y1, x2, y2 = map(int, p.xyxy[0])

            person_has_id = False

            # Check if ID box inside person box
            for i in id_cards:
                ix1, iy1, ix2, iy2 = map(int, i.xyxy[0])

                # ID card center point
                cx = (ix1 + ix2) // 2
                cy = (iy1 + iy2) // 2

                if x1 < cx < x2 and y1 < cy < y2:
                    person_has_id = True

            if person_has_id:
                color = (0, 255, 0)
                label = "ID CARD PRESENT"
            else:
                color = (0, 0, 255)
                label = "NO ID CARD"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Person + ID Check", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
