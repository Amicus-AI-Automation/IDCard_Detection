import cv2
import os
import re
import time
from ultralytics import YOLO
from deepface import DeepFace
import easyocr
from fuzzywuzzy import fuzz

# ---------------- CONFIG ----------------
CONF_THRESHOLD = 0.5
KNOWN_FACES_DIR = "known_faces"
RESET_TIME = 3  # seconds

# ---------------- LOAD MODELS ----------------
person_model = YOLO("yolov8n.pt")
id_model = YOLO("runs/detect/retrain_v2/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- STATE ----------------
person_state = {}
last_seen_id = {}

# ---------------- FACE RECOGNITION ----------------
def recognize_face(face_img):
    try:
        result = DeepFace.find(
            img_path=face_img,
            db_path=KNOWN_FACES_DIR,
            enforce_detection=False,
            model_name="Facenet"
        )

        if len(result) > 0 and not result[0].empty:
            path = result[0].iloc[0]['identity']
            return path.split(os.sep)[-2]
    except:
        pass

    return "Unknown"

# ---------------- OCR ----------------
def extract_name(id_crop):

    id_crop = cv2.resize(id_crop, None, fx=2.5, fy=2.5,
                         interpolation=cv2.INTER_LINEAR)

    results = reader.readtext(id_crop)

    print("\n OCR DEBUG:")

    name = ""
    for (_, text, prob) in results:
        print(text, f"{prob:.2f}")

        if prob > 0.3:
            clean = text.strip()

            if clean.lower() in ["amicus", "amicius"]:
                continue

            if re.match(r'^[A-Za-z ]+$', clean):
                if 4 < len(clean) < 30:
                    name += clean + " "

    return name.strip()

# ---------------- FUZZY MATCH ----------------
def match_names(face_name, id_name):

    if not id_name or not face_name or face_name == "Unknown":
        return False

    score_full = fuzz.ratio(face_name.lower(), id_name.lower())
    score_partial = fuzz.partial_ratio(face_name.lower(), id_name.lower())

    final_score = max(score_full, score_partial)

    print(f" MATCH → Full: {score_full}, Partial: {score_partial}, Final: {final_score}")

    return final_score > 75

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    current_time = time.time()

    person_results = person_model(frame, conf=CONF_THRESHOLD, verbose=False)
    id_results = id_model(frame, conf=CONF_THRESHOLD, verbose=False)

    persons = person_results[0].boxes
    ids = id_results[0].boxes

    for p in persons:

        if int(p.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(int, p.xyxy[0])
        box_key = (x1//50, y1//50)

        # Initialize
        if box_key not in person_state:
            person_state[box_key] = {
                "status": "NO_ID",
                "name": "",
                "locked": False
            }

        state = person_state[box_key]

        # ---------------- CHECK ID ----------------
        has_id = False
        id_crop = None

        for i in ids:
            ix1, iy1, ix2, iy2 = map(int, i.xyxy[0])

            cx = (ix1 + ix2)//2
            cy = (iy1 + iy2)//2

            if x1 < cx < x2 and y1 < cy < y2:
                has_id = True

                pad = 40
                ix1 = max(0, ix1-pad)
                iy1 = max(0, iy1-pad)
                ix2 = min(frame.shape[1], ix2+pad)
                iy2 = min(frame.shape[0], iy2+pad)

                id_crop = frame[iy1:iy2, ix1:ix2]
                break

        # ---------------- RESET LOGIC ----------------
        if not has_id:
            if box_key in last_seen_id:
                if current_time - last_seen_id[box_key] > RESET_TIME:
                    person_state[box_key] = {
                        "status": "NO_ID",
                        "name": "",
                        "locked": False
                    }
        else:
            last_seen_id[box_key] = current_time

        # ---------------- MAIN LOGIC ----------------
        if has_id:

            #  LOCKED → skip everything
            if state["locked"]:
                label = state["name"] + " +"
                color = (0, 255, 0)

            else:
                #  ID detected
                state["status"] = "ID_DETECTED"

                # -------- FACE --------
                face_crop = frame[y1:y1+(y2-y1)//2, x1:x2]
                face_name = recognize_face(face_crop)

                # -------- OCR --------
                id_name = extract_name(id_crop) if id_crop is not None else ""

                # -------- MATCH --------
                if id_name and face_name != "Unknown":

                    if match_names(face_name, id_name):

                        state["status"] = "AUTHENTICATED"
                        state["name"] = face_name
                        state["locked"] = True

                        label = face_name + " +"
                        color = (0, 255, 0)

                    else:
                        state["status"] = "FAILED"
                        label = face_name + " -"
                        color = (255, 0, 255) 

                else:
                    label = face_name
                    color = (0, 255, 255)

        else:
            label = "NO ID"
            color = (0, 0, 255)

        # ---------------- DRAW ----------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

    cv2.imshow("Final Identity System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
