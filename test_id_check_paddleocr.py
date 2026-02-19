# -------------------------------
# ENV FIXES FOR PADDLE 3.x
# -------------------------------
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"   # disable oneDNN
os.environ["FLAGS_use_onednn"] = "0"

# -------------------------------
# IMPORTS
# -------------------------------
import cv2
import re
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ---------------------------------------
# CONFIG
# ---------------------------------------
IMAGE_PATH = "test_id.jpg"
DEBUG_DIR = "debug_outputs_paddleocr"
os.makedirs(DEBUG_DIR, exist_ok=True)

# ---------------------------------------
# LOAD MODELS
# ---------------------------------------
person_model = YOLO("yolov8n.pt")
id_model = YOLO("runs/detect/retrain_v2/weights/best.pt")

# Use lighter mobile models (more stable)
ocr = PaddleOCR(
    lang='en',
    det_model_dir=None,
    rec_model_dir=None
)

# ---------------------------------------
# LOAD IMAGE
# ---------------------------------------
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print("Image not found")
    exit()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------------------
# DETECT PERSON + ID
# ---------------------------------------
person_results = person_model(frame, conf=0.5)
id_results = id_model(frame, conf=0.5)

persons = person_results[0].boxes
id_boxes = id_results[0].boxes

print("Persons detected:", len(persons))
print("ID detections:", len(id_boxes))

for p in persons:

    if int(p.cls[0]) != 0:
        continue

    x1, y1, x2, y2 = map(int, p.xyxy[0])

    # Save person debug
    person_debug = frame.copy()
    cv2.rectangle(person_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(f"{DEBUG_DIR}/person_{timestamp}.jpg", person_debug)

    strap_box = None

    # Confirm ID inside person
    for box in id_boxes:
        ix1, iy1, ix2, iy2 = map(int, box.xyxy[0])
        cx = (ix1 + ix2) // 2
        cy = (iy1 + iy2) // 2

        if x1 < cx < x2 and y1 < cy < y2:
            strap_box = box
            break

    if strap_box is None:
        print("No ID worn")
        continue

    ix1, iy1, ix2, iy2 = map(int, strap_box.xyxy[0])

    strap_debug = frame.copy()
    cv2.rectangle(strap_debug, (ix1, iy1), (ix2, iy2), (255, 0, 0), 2)
    cv2.imwrite(f"{DEBUG_DIR}/strap_{timestamp}.jpg", strap_debug)

    # Crop strap region
    strap_region = frame[iy1:iy2, ix1:ix2]

    # Enhance
    strap_region = cv2.resize(
        strap_region, None,
        fx=2.0, fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    lab = cv2.cvtColor(strap_region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    strap_region = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    cv2.imwrite(f"{DEBUG_DIR}/enhanced_strap_{timestamp}.jpg", strap_region)

    # ---------------------------------------
    # PADDLE OCR (NEW API)
    # ---------------------------------------
    result = ocr.predict(strap_region)

    print("\nOCR OUTPUT:")
    detected_id = ""
    detected_name = ""

    for res in result:
        for line in res["rec_texts"]:
            text = line
            print(text)

            # Extract ID
            match = re.search(r'\d{4,8}', text)
            if match:
                detected_id = match.group()

            # Extract Name
            if re.match(r'^[A-Za-z ]+$', text):
                if len(text) > 3 and text.lower() not in ["amicus"]:
                    detected_name += text + " "

    detected_name = detected_name.strip()

    print("\nFINAL RESULT")
    print("ID:", detected_id)
    print("Name:", detected_name)

    # Draw output
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if detected_id:
        cv2.putText(frame, f"ID: {detected_id}",
                    (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

    if detected_name:
        cv2.putText(frame, f"Name: {detected_name}",
                    (x1, y2 + 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)

# Save final
final_path = f"{DEBUG_DIR}/final_{timestamp}.jpg"
cv2.imwrite(final_path, frame)

print("\nSaved debug images in:", DEBUG_DIR)

cv2.imshow("Final Result - PaddleOCR", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
