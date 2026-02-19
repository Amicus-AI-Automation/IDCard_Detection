import cv2
from ultralytics import YOLO
import easyocr
import re
import numpy as np

# ---------------------------------------
# LOAD MODELS
# ---------------------------------------
person_model = YOLO("yolov8n.pt")
id_model = YOLO("runs/detect/retrain_v2/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

# ---------------------------------------
# CAMERA
# ---------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("🎥 Live feed started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    person_results = person_model(frame, conf=0.5, verbose=False)
    id_results = id_model(frame, conf=0.5, verbose=False)

    persons = person_results[0].boxes
    ids = id_results[0].boxes

    for p in persons:

        if int(p.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(int, p.xyxy[0])

        for i in ids:

            ix1, iy1, ix2, iy2 = map(int, i.xyxy[0])

            cx = (ix1 + ix2) // 2
            cy = (iy1 + iy2) // 2

            if x1 < cx < x2 and y1 < cy < y2:

                # ---- Crop ID ----
                id_crop = frame[iy1:iy2, ix1:ix2]

                h, w = id_crop.shape[:2]

                # ---- Split Regions ----
                name_region = id_crop[int(0.35*h):int(0.65*h), :]
                id_region   = id_crop[int(0.65*h):int(0.85*h), :]

                # ---- Preprocess Function ----
                def preprocess(img):
                    img = cv2.resize(img, None, fx=4, fy=4,
                                     interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 120, 255,
                                              cv2.THRESH_BINARY)
                    return thresh

                name_region = preprocess(name_region)
                id_region = preprocess(id_region)

                # ---- OCR ----
                name_text = reader.readtext(name_region)
                id_text = reader.readtext(id_region)

                detected_name = ""
                detected_id = ""

                for (_, text, prob) in name_text:
                    if prob > 0.4:
                        clean = text.strip()
                        if re.match(r'^[A-Za-z ]+$', clean):
                            if len(clean) > 3:
                                detected_name += clean + " "

                for (_, text, prob) in id_text:
                    if prob > 0.4:
                        match = re.search(r'\d{4,8}', text)
                        if match:
                            detected_id = match.group()

                detected_name = detected_name.strip()

                # ---- Draw Output ----
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                if detected_name:
                    cv2.putText(frame,
                                detected_name,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0,255,0),
                                2)

                if detected_id:
                    cv2.putText(frame,
                                f"ID: {detected_id}",
                                (x1, y2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0,255,255),
                                2)

    cv2.imshow("Live ID Detection + OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
