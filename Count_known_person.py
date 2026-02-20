import cv2
import csv
import os
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace

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

# ---------------- LOAD MODELS ----------------
person_model = YOLO(PERSON_MODEL_PATH)
id_model = YOLO(ID_MODEL_PATH)
tracker = DeepSort(max_age=60, n_init=2, max_cosine_distance=0.3, nn_budget=100)

# ---------------- STORAGE ----------------
people_data = {}
inside_state = {}
last_seen_frame = {}
id_history = {}
trackid_to_identity = {}
frame_index = 0

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", buffering=1) as f:
        writer = csv.writer(f)
        writer.writerow([
            "person_id",
            "name",
            "in_count",
            "out_count",
            "is_wearing_id",
            "first_entry_time",
            "last_exit_time"
        ])

# ---------------- HELPERS ----------------
def save_to_csv():
    rows = []
    for pid, info in people_data.items():
        rows.append([
            pid,
            info["name"],
            info["in_count"],
            info["out_count"],
            info["is_wearing_id"],
            info["first_entry_time"],
            info["last_exit_time"]
        ])
    df = pd.DataFrame(rows, columns=[
        "person_id", "name", "in_count", "out_count",
        "is_wearing_id", "first_entry_time", "last_exit_time"
    ])
    df.to_csv(CSV_PATH, index=False)
    print("📁 CSV Updated")

def detect_id_on_person(frame, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    results = id_model(crop, conf=0.4, verbose=False)[0]
    return len(results.boxes) > 0

def recognize_face(face_crop):
    try:
        result = DeepFace.find(
            img_path=face_crop,
            db_path=KNOWN_FACES_DIR,
            enforce_detection=False,
            model_name="Facenet",
            silent=True
        )
        if len(result) > 0 and not result[0].empty:
            identity_path = result[0].iloc[0]['identity']
            return identity_path.split(os.sep)[-2]
    except:
        pass
    return "Unknown"

# ---------------- MAIN LOOP ----------------
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

print("🎥 Camera started... Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1

    results = person_model(frame, conf=0.5, verbose=False)[0]
    detections = []

    for box in results.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            detections.append(([float(x1), float(y1), float(x2-x1), float(y2-y1)], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)
    active_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        pid = track.track_id
        l, t, w, h = map(int, track.to_ltrb())
        x1, y1, x2, y2 = l, t, l+w, t+h

        active_ids.add(pid)
        last_seen_frame[pid] = frame_index

        if pid not in people_data:
            face_crop = frame[y1:y1+(y2-y1)//2, x1:x2]
            name = recognize_face(face_crop) if face_crop.size != 0 else "Unknown"

            people_data[pid] = {
                "name": name,
                "in_count": 1,
                "out_count": 0,
                "is_wearing_id": False,
                "first_entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_exit_time": ""
            }
            inside_state[pid] = True
            id_history[pid] = deque(maxlen=15)
            print(f"➡️ {name} ENTERED")

        has_id = detect_id_on_person(frame, (x1, y1, x2, y2))
        id_history[pid].append(has_id)

        if sum(id_history[pid]) >= ID_CONFIRM_FRAMES:
            people_data[pid]["is_wearing_id"] = True

        color = (0, 255, 0) if people_data[pid]["is_wearing_id"] else (0, 0, 255)
        id_text = "ID OK" if people_data[pid]["is_wearing_id"] else "NO ID"

        label = f"{people_data[pid]['name']} | IN:{people_data[pid]['in_count']} OUT:{people_data[pid]['out_count']} | {id_text}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    for pid in list(inside_state.keys()):
        if pid not in active_ids:
            if frame_index - last_seen_frame.get(pid, 0) > LOST_FRAME_THRESHOLD:
                people_data[pid]["out_count"] += 1
                people_data[pid]["last_exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"⬅️ {people_data[pid]['name']} EXITED")
                inside_state.pop(pid)
                save_to_csv()

    cv2.imshow("Office Entry + Face + ID Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
save_to_csv()
print("✅ Final CSV saved:", CSV_PATH)
