import cv2
from ultralytics import YOLO
import easyocr
import re
import os
from datetime import datetime

# ---------------------------------------
# CONFIG
# ---------------------------------------
BASE_DEBUG_DIR = "debug_outputs"

session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
DEBUG_DIR = os.path.join(BASE_DEBUG_DIR, f"live_{session_time}")
os.makedirs(DEBUG_DIR, exist_ok=True)

# ---------------------------------------
# Load Models
# ---------------------------------------
person_model = YOLO("yolov8n.pt")
id_model = YOLO("runs/detect/retrain_v2/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

# ---------------------------------------
# Start Camera
# ---------------------------------------
cap = cv2.VideoCapture(0)

frame_count = 0

print("🎥 Live feed started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    person_results = person_model(frame, conf=0.5)
    id_results = id_model(frame, conf=0.5)

    persons = person_results[0].boxes
    id_cards = id_results[0].boxes

    for p in persons:
        cls = int(p.cls[0])
        if cls != 0:
            continue

        x1, y1, x2, y2 = map(int, p.xyxy[0])

        person_has_id = False
        detected_id = ""
        detected_name = ""

        for i in id_cards:
            ix1, iy1, ix2, iy2 = map(int, i.xyxy[0])

            cx = (ix1 + ix2) // 2
            cy = (iy1 + iy2) // 2

            # Check if ID inside person box
            if x1 < cx < x2 and y1 < cy < y2:
                person_has_id = True

                # Add padding
                padding = 20
                ix1 = max(0, ix1 - padding)
                iy1 = max(0, iy1 - padding)
                ix2 = min(frame.shape[1], ix2 + padding)
                iy2 = min(frame.shape[0], iy2 + padding)

                id_crop = frame[iy1:iy2, ix1:ix2]

                # Resize for OCR
                id_crop = cv2.resize(
                    id_crop, None, fx=2, fy=2,
                    interpolation=cv2.INTER_CUBIC
                )

                # Run OCR
                ocr_results = reader.readtext(id_crop)

                for (bbox, text, prob) in ocr_results:
                    if prob > 0.3:

                        # Extract ID number
                        match = re.search(r'\d{4,8}', text)
                        if match:
                            detected_id = match.group()

                        # Extract Name
                        clean = text.strip()
                        if re.match(r'^[A-Za-z ]+$', clean):
                            if len(clean) > 3 and clean.lower() not in ["amicus"]:
                                detected_name += clean + " "

                detected_name = detected_name.strip()

                # Save debug only when ID detected
                if detected_id or detected_name:
                    cv2.imwrite(
                        f"{DEBUG_DIR}/frame_{frame_count}.jpg",
                        frame
                    )
                    cv2.imwrite(
                        f"{DEBUG_DIR}/id_crop_{frame_count}.jpg",
                        id_crop
                    )

        # Draw result
        if person_has_id:
            color = (0, 255, 0)
            label = "ID PRESENT"
        else:
            color = (0, 0, 255)
            label = "NO ID"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        if detected_id:
            cv2.putText(frame, f"ID: {detected_id}",
                        (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,255), 2)

        if detected_name:
            cv2.putText(frame, f"Name: {detected_name}",
                        (x1, y2 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255,255,0), 2)

    cv2.imshow("Live ID Detection + OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n📁 Live session saved in: {DEBUG_DIR}")
