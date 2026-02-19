import cv2
from ultralytics import YOLO
import easyocr
import re
import os
from datetime import datetime

# ---------------------------------------
# CONFIG
# ---------------------------------------
BASE_DEBUG_DIR = "debug_live"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DEBUG_DIR = os.path.join(BASE_DEBUG_DIR, f"session_{timestamp}")
os.makedirs(DEBUG_DIR, exist_ok=True)

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

print(f"🎥 Live feed started. Debug saving to: {DEBUG_DIR}")
print("Press 'q' to exit.")

# ---------------------------------------
# STATE
# ---------------------------------------
frame_count = 0

# Per-person name cache
person_names = {}

# ---------------------------------------
# LOOP
# ---------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    person_results = person_model(frame, conf=0.5, verbose=False)
    id_results = id_model(frame, conf=0.5, verbose=False)

    persons = person_results[0].boxes
    ids = id_results[0].boxes

    for p_idx, p in enumerate(persons):

        if int(p.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(int, p.xyxy[0])

        id_found_for_person = False
        detected_name = ""

        for jdx, i in enumerate(ids):

            ix1, iy1, ix2, iy2 = map(int, i.xyxy[0])

            # Center check (ID belongs to person)
            cx = (ix1 + ix2) // 2
            cy = (iy1 + iy2) // 2

            if x1 < cx < x2 and y1 < cy < y2:

                id_found_for_person = True

                # ---------------------------------------
                # PADDING
                # ---------------------------------------
                pad = 40
                ix1_p = max(0, ix1 - pad)
                iy1_p = max(0, iy1 - pad)
                ix2_p = min(frame.shape[1], ix2 + pad)
                iy2_p = min(frame.shape[0], iy2 + pad)

                id_crop = frame[iy1_p:iy2_p, ix1_p:ix2_p]

                # ---------------------------------------
                # SAFE UPSCALE
                # ---------------------------------------
                id_crop = cv2.resize(
                    id_crop,
                    None,
                    fx=2.5,
                    fy=2.5,
                    interpolation=cv2.INTER_LINEAR
                )

                # ---------------------------------------
                # OCR EVERY 15 FRAMES
                # ---------------------------------------
                if frame_count % 15 == 0:

                    ocr_results = reader.readtext(id_crop)

                    print(f"\n🔍 OCR DEBUG (Person {p_idx}):")

                    for (_, text, prob) in ocr_results:
                        print(text, f"{prob:.2f}")

                        if prob > 0.25:
                            clean = text.strip()

                            # Ignore noise
                            if clean.lower() in ["amicius", "amicus"]:
                                continue

                            # Name filter
                            if re.match(r'^[A-Za-z ]+$', clean):
                                if 4 < len(clean) < 30:
                                    detected_name += clean + " "

                    detected_name = detected_name.strip()

                    # Save valid name per person
                    if detected_name:
                        person_names[p_idx] = detected_name

                    # ---------------------------------------
                    # SAVE DEBUG IMAGES
                    # ---------------------------------------
                    base_name = f"{timestamp}_frame{frame_count}_p{p_idx}_id{jdx}"

                    debug_frame = frame.copy()
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.rectangle(debug_frame, (ix1, iy1), (ix2, iy2), (255,0,0), 2)

                    cv2.imwrite(f"{DEBUG_DIR}/{base_name}_frame.jpg", debug_frame)
                    cv2.imwrite(f"{DEBUG_DIR}/{base_name}_crop.jpg", id_crop)

        # ---------------------------------------
        # DRAW OUTPUT PER PERSON
        # ---------------------------------------
        if id_found_for_person:

            if p_idx in person_names:
                # ✅ Name detected
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                cv2.putText(frame,
                            person_names[p_idx],
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0,255,0),
                            2)

            else:
                # 🟡 ID present but name not read yet
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)

                cv2.putText(frame,
                            "ID Detected",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0,255,255),
                            2)

        else:
            # ❌ No ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)

            cv2.putText(frame,
                        "NO ID",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0,0,255),
                        2)

    # ---------------------------------------
    # DISPLAY
    # ---------------------------------------
    cv2.imshow("Live ID + Name Detection (Multi-Person)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------------------
# CLEANUP
# ---------------------------------------
cap.release()
cv2.destroyAllWindows()

print(f"\n📁 Debug images saved in: {DEBUG_DIR}")
