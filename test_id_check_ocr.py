import cv2
from ultralytics import YOLO
import easyocr
import re
import os
from datetime import datetime
import numpy as np

# ---------------------------------------
# DEBUG SESSION FOLDER
# ---------------------------------------
BASE_DEBUG_DIR = "debug_outputs"
session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
DEBUG_DIR = os.path.join(BASE_DEBUG_DIR, f"live_{session_time}")
os.makedirs(DEBUG_DIR, exist_ok=True)

# ---------------------------------------
# LOAD MODELS
# ---------------------------------------
person_model = YOLO("yolov8n.pt")
id_model = YOLO("runs/detect/retrain_v2/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

# ---------------------------------------
# START CAMERA
# ---------------------------------------
cap = cv2.VideoCapture(0)

# Increase webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0

print("🎥 Live feed started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    person_results = person_model(frame, conf=0.5)
    id_results = id_model(frame, conf=0.3)  # Lower threshold

    persons = person_results[0].boxes
    id_cards = id_results[0].boxes

    for p in persons:

        if int(p.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(int, p.xyxy[0])

        detected_id = ""
        detected_name = ""
        person_has_id = False

        for i in id_cards:

            ix1, iy1, ix2, iy2 = map(int, i.xyxy[0])
            cx = (ix1 + ix2) // 2
            cy = (iy1 + iy2) // 2

            # Check ID inside person box
            if x1 < cx < x2 and y1 < cy < y2:
                person_has_id = True

                # Add padding
                padding = 15
                ix1 = max(0, ix1 - padding)
                iy1 = max(0, iy1 - padding)
                ix2 = min(frame.shape[1], ix2 + padding)
                iy2 = min(frame.shape[0], iy2 + padding)

                id_crop_full = frame[iy1:iy2, ix1:ix2]

                # ---------------------------------------
                # FOCUS ON LOWER TEXT REGION
                # ---------------------------------------
                h = id_crop_full.shape[0]
                text_region = id_crop_full[int(0.35*h):h, :]

                # Aggressive upscale
                text_region = cv2.resize(
                    text_region,
                    None,
                    fx=4,
                    fy=4,
                    interpolation=cv2.INTER_CUBIC
                )

                # Convert to grayscale
                gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)

                # Sharpen
                gray = cv2.GaussianBlur(gray, (3,3), 0)
                gray = cv2.addWeighted(gray, 1.5, gray, -0.5, 0)

                # Threshold
                _, thresh = cv2.threshold(
                    gray,
                    120,
                    255,
                    cv2.THRESH_BINARY
                )

                # Save debug crop
                cv2.imwrite(
                    f"{DEBUG_DIR}/id_crop_{frame_count}.jpg",
                    thresh
                )

                # ---------------------------------------
                # OCR
                # ---------------------------------------
                ocr_results = reader.readtext(thresh)

                print("\nOCR OUTPUT:")
                for (bbox, text, prob) in ocr_results:
                    print(text, prob)

                    if prob > 0.3:

                        # Extract ID number
                        match = re.search(r'\d{4,8}', text)
                        if match:
                            detected_id = match.group()

                        # Extract Name (flexible)
                        clean = text.strip()
                        if re.search(r'[A-Za-z]{3,}', clean):
                            if clean.lower() not in ["amicus"]:
                                detected_name += clean + " "

                detected_name = detected_name.strip()

        # ---------------------------------------
        # DRAW OUTPUT
        # ---------------------------------------
        if person_has_id:
            color = (0, 255, 0)
            label = "ID PRESENT"
        else:
            color = (0, 0, 255)
            label = "NO ID"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Prepare display text
        display_text = ""

        if detected_name:
            display_text += detected_name

        if detected_id:
            if display_text:
                display_text += f" | {detected_id}"
            else:
                display_text = detected_id

        if not display_text:
            display_text = label

        # Draw background rectangle
        (text_w, text_h), _ = cv2.getTextSize(
            display_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        cv2.rectangle(
            frame,
            (x1, y1 - text_h - 15),
            (x1 + text_w + 10, y1),
            color,
            -1
        )

        cv2.putText(
            frame,
            display_text,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

    cv2.imshow("Live ID Detection + OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n📁 Debug session saved in: {DEBUG_DIR}")
