import cv2
from ultralytics import YOLO
import easyocr
import re
import os
from datetime import datetime

# ---------------------------------------
# CONFIG
# ---------------------------------------
IMAGE_PATH = "Screenshot 2026-02-18 193831.jpg"  # your image
BASE_DEBUG_DIR = "debug_outputs"

# Extract image name without extension
image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

# Create per-image debug directory
DEBUG_DIR = os.path.join(BASE_DEBUG_DIR, image_name)

os.makedirs(DEBUG_DIR, exist_ok=True)

# ---------------------------------------
# Load Models
# ---------------------------------------
person_model = YOLO("yolov8n.pt")
id_model = YOLO("runs/detect/retrain_v2/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

# ---------------------------------------
# Load Image
# ---------------------------------------
frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print("❌ Image not found!")
    exit()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------------------
# Detect Person + ID
# ---------------------------------------
person_results = person_model(frame, conf=0.5)
id_results = id_model(frame, conf=0.5)

persons = person_results[0].boxes
id_cards = id_results[0].boxes

print(f"Detected {len(persons)} person(s)")
print(f"Detected {len(id_cards)} ID card(s)")

# ---------------------------------------
# Process Persons
# ---------------------------------------
for idx, p in enumerate(persons):

    cls = int(p.cls[0])
    if cls != 0:
        continue

    x1, y1, x2, y2 = map(int, p.xyxy[0])

    # Draw & Save Person Box
    person_image = frame.copy()
    cv2.rectangle(person_image, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.imwrite(f"{DEBUG_DIR}/person_{timestamp}_{idx}.jpg", person_image)

    # ---------------------------------------
    # Check if ID belongs to this person
    # ---------------------------------------
    for jdx, i in enumerate(id_cards):

        ix1, iy1, ix2, iy2 = map(int, i.xyxy[0])

        cx = (ix1 + ix2) // 2
        cy = (iy1 + iy2) // 2

        if x1 < cx < x2 and y1 < cy < y2:

            # Draw & Save ID Detection
            id_image = frame.copy()
            cv2.rectangle(id_image, (ix1, iy1), (ix2, iy2), (255,0,0), 2)
            cv2.imwrite(f"{DEBUG_DIR}/id_{timestamp}_{jdx}.jpg", id_image)

            # ---------------------------------------
            # Add Padding
            # ---------------------------------------
            padding = 20
            ix1 = max(0, ix1 - padding)
            iy1 = max(0, iy1 - padding)
            ix2 = min(frame.shape[1], ix2 + padding)
            iy2 = min(frame.shape[0], iy2 + padding)

            id_crop = frame[iy1:iy2, ix1:ix2]

            # Resize for better OCR
            id_crop = cv2.resize(
                id_crop, None, fx=2, fy=2,
                interpolation=cv2.INTER_CUBIC
            )

            # Save Cropped ID
            cv2.imwrite(f"{DEBUG_DIR}/id_crop_{timestamp}_{jdx}.jpg", id_crop)

            # ---------------------------------------
            # OCR Full ID
            # ---------------------------------------
            ocr_results = reader.readtext(id_crop)

            print("\nRAW OCR OUTPUT:")
            for r in ocr_results:
                print(r[1], r[2])

            detected_id = ""
            detected_name = ""

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

            print("\n✅ FINAL RESULT")
            print("ID Number:", detected_id)
            print("Name:", detected_name)

            # ---------------------------------------
            # Draw Final Output
            # ---------------------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            if detected_id:
                cv2.putText(frame, f"ID: {detected_id}",
                            (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,255,255), 2)

            if detected_name:
                cv2.putText(frame, f"Name: {detected_name}",
                            (x1, y2 + 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255,255,0), 2)

# ---------------------------------------
# Save Final Output Image
# ---------------------------------------
final_path = f"{DEBUG_DIR}/final_output_{timestamp}.jpg"
cv2.imwrite(final_path, frame)

print(f"\n📁 All debug images saved in: {DEBUG_DIR}")
print(f"🖼 Final image saved as: {final_path}")

cv2.imshow("Final Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
