import cv2
import csv
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
import os
from collections import deque

# ------------------ CONFIG ------------------
PERSON_MODEL_PATH = "yolov8n.pt"
ID_MODEL_PATH = "runs/detect/retrain_v2/weights/best.pt"
CSV_PATH = "data/entry_exit_log.csv"
CAMERA_INDEX = 0

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
LOST_FRAME_THRESHOLD = 20
ID_CONFIRM_FRAMES = 5

# ------------------ LOAD MODELS ------------------
person_model = YOLO(PERSON_MODEL_PATH)
id_model = YOLO(ID_MODEL_PATH)

tracker = DeepSort(max_age=60, n_init=2, max_cosine_distance=0.3, nn_budget=100)

# ------------------ STORAGE ------------------
people_data = {}
inside_state = {}
last_seen_frame = {}
id_history = {}   # person_id -> deque of last N frames
frame_index = 0

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", buffering=1) as f:
        writer = csv.writer(f)
        writer.writerow([
            "person_id",
            "in_count",
            "out_count",
            "is_wearing_id",
            "first_entry_time",
            "last_exit_time"
        ])

# ------------------ HELPERS ------------------
def save_to_csv():
    rows = []
    for pid, info in people_data.items():
        rows.append([
            pid,
            info["in_count"],
            info["out_count"],
            info["is_wearing_id"],
            info["first_entry_time"],
            info["last_exit_time"]
        ])

    df = pd.DataFrame(rows, columns=[
        "person_id", "in_count", "out_count",
        "is_wearing_id", "first_entry_time", "last_exit_time"
    ])
    df.to_csv(CSV_PATH, index=False)

def detect_id_on_person(frame, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    results = id_model(crop, conf=0.4, verbose=False)[0]
    return len(results.boxes) > 0

# ------------------ MAIN LOOP ------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

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
            detections.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)
    active_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        pid = track.track_id
        l, t, w, h = map(int, track.to_ltrb())
        x1, y1, x2, y2 = l, t, l + w, t + h

        active_ids.add(pid)
        last_seen_frame[pid] = frame_index

        if pid not in people_data:
            people_data[pid] = {
                "in_count": 1,
                "out_count": 0,
                "is_wearing_id": False,
                "first_entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_exit_time": ""
            }
            inside_state[pid] = True
            id_history[pid] = deque(maxlen=15)

        has_id = detect_id_on_person(frame, (x1, y1, x2, y2))
        id_history[pid].append(has_id)

        # ID smoothing logic
        if sum(id_history[pid]) >= ID_CONFIRM_FRAMES:
            people_data[pid]["is_wearing_id"] = True

        # Color logic
        color = (0, 255, 0) if people_data[pid]["is_wearing_id"] else (0, 0, 255)

        label = f"In: {people_data[pid]['in_count']} | Out: {people_data[pid]['out_count']} | ID: {'Yes' if people_data[pid]['is_wearing_id'] else 'No'}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Person {pid}", (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, label, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Exit detection (only when missing for N frames)
    for pid in list(inside_state.keys()):
        if pid not in active_ids:
            if frame_index - last_seen_frame.get(pid, 0) > LOST_FRAME_THRESHOLD:
                people_data[pid]["out_count"] += 1
                people_data[pid]["last_exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                inside_state.pop(pid)
                save_to_csv()

    cv2.imshow("Office Entry Exit Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
