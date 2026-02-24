import cv2
import csv
import os
import re
import time
from datetime import datetime
from collections import deque
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace
import easyocr
from fuzzywuzzy import fuzz

# ---------------- CONFIG ----------------
PERSON_MODEL_PATH = "yolov8n.pt"
ID_MODEL_PATH = "runs/detect/retrain_v2/weights/best.pt"
KNOWN_FACES_DIR = "known_faces"
CSV_PATH = "data/entry_exit_log.csv"
CAMERA_INDEX = 0

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
LOST_FRAME_THRESHOLD = 20
ID_CONFIRM_FRAMES = 5
RESET_TIME = 3

# ---------------- LOAD MODELS ----------------
person_model = YOLO(PERSON_MODEL_PATH)
id_model = YOLO(ID_MODEL_PATH)
tracker = DeepSort(max_age=60, n_init=2, max_cosine_distance=0.3, nn_budget=100)
reader = easyocr.Reader(['en'], gpu=False)

# ---------------- STORAGE ----------------
people_data = {}
active_tracks = {}
inside_people = set()
last_seen_frame = {}
id_history = {}
person_state = {}
last_seen_id = {}

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", buffering=1) as f:
        writer = csv.writer(f)
        writer.writerow(["name", "in_count", "out_count", "is_wearing_id", "first_entry_time", "last_exit_time"])

# ---------------- HELPERS ----------------
def save_to_csv():
    rows = []
    for name, info in people_data.items():
        rows.append([
            name,
            info["in_count"],
            info["out_count"],
            info["is_wearing_id"],
            info["first_entry_time"],
            info["last_exit_time"]
        ])
    pd.DataFrame(rows, columns=[
        "name", "in_count", "out_count",
        "is_wearing_id", "first_entry_time", "last_exit_time"
    ]).to_csv(CSV_PATH, index=False)

def detect_id_on_person(frame, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, None

    results = id_model(crop, conf=0.4, verbose=False)[0].boxes
    if len(results) == 0:
        return False, None

    b = results[0].xyxy[0].cpu().numpy().astype(int)
    ix1, iy1, ix2, iy2 = b
    id_crop = crop[iy1:iy2, ix1:ix2]
    return True, id_crop

def recognize_face(face_crop):
    try:
        result = DeepFace.find(face_crop, db_path=KNOWN_FACES_DIR, enforce_detection=False, silent=True)
        if len(result) > 0 and len(result[0]) > 0:
            return os.path.basename(os.path.dirname(result[0].iloc[0]["identity"]))
    except:
        pass
    return "Unknown"

def extract_name(id_crop):
    id_crop = cv2.resize(id_crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
    results = reader.readtext(id_crop)
    name = ""
    for (_, text, prob) in results:
        if prob > 0.3:
            clean = text.strip()
            if re.match(r'^[A-Za-z ]+$', clean) and 4 < len(clean) < 30:
                name += clean + " "
    return name.strip()

def match_names(face_name, id_name):
    if not id_name or not face_name or face_name == "Unknown":
        return False

    score = max(
        fuzz.ratio(face_name.lower(), id_name.lower()),
        fuzz.partial_ratio(face_name.lower(), id_name.lower())
    )
    return score > 75

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

print("🎥 Camera started... Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = person_model(frame, conf=0.5, verbose=False)[0]
    detections = []

    for box in results.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            detections.append(([float(x1), float(y1), float(x2-x1), float(y2-y1)], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)
    current_inside = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        pid = track.track_id
        l, t, w, h = map(int, track.to_ltrb())
        x1, y1, x2, y2 = l, t, l+w, t+h

        face_crop = frame[y1:y1+(y2-y1)//2, x1:x2]
        name = recognize_face(face_crop)

        active_tracks[pid] = name
        current_inside.add(name)

        if name not in people_data:
            people_data[name] = {
                "in_count": 1,
                "out_count": 0,
                "is_wearing_id": False,
                "first_entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_exit_time": ""
            }
        elif name not in inside_people:
            people_data[name]["in_count"] += 1

        inside_people.add(name)

        has_id, id_crop = detect_id_on_person(frame, (x1, y1, x2, y2))

        label = name
        color = (0, 0, 255)

        if has_id:
            id_name = extract_name(id_crop) if id_crop is not None else ""
            if match_names(name, id_name):
                label = f"{name} +"
                color = (0, 255, 0)
            else:
                label = f"{name} -"
                color = (255, 0, 255)
        else:
            label = "NO ID"
            color = (0, 0, 255)

        stats = f"IN:{people_data[name]['in_count']} OUT:{people_data[name]['out_count']}"

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{label} | {stats}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for name in list(inside_people):
        if name not in current_inside:
            people_data[name]["out_count"] += 1
            people_data[name]["last_exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            inside_people.remove(name)
            save_to_csv()

    cv2.imshow("Unified Office Security System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
save_to_csv()
print("✅ Final CSV saved:", CSV_PATH)