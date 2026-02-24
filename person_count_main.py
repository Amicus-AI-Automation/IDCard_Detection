import cv2
import csv
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
import os

# ------------------ CONFIG ------------------
PERSON_MODEL_PATH = "yolov8n.pt"
ID_MODEL_PATH = "runs/detect/retrain_v2/weights/best.pt"
CSV_PATH = "data/entry_exit_log1.csv"
CAMERA_INDEX = 0
FRAME_LOST_THRESHOLD = 30

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# ------------------ LOAD MODELS ------------------
person_model = YOLO(PERSON_MODEL_PATH)
id_model = YOLO(ID_MODEL_PATH)

tracker = DeepSort(max_age=60, n_init=3, max_cosine_distance=0.3, nn_budget=100)

# ------------------ STORAGE ------------------
people_data = {}
last_seen = {}
trackid_to_personid = {}
next_person_id = 1

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", buffering=1) as f:
        writer = csv.writer(f)
        writer.writerow(["person_id", "in_count", "out_count", "is_wearing_id"])

# ------------------ HELPERS ------------------
def save_to_csv(data_dict):
    rows = []
    for pid, info in data_dict.items():
        rows.append([pid, info["in_count"], info["out_count"], info["is_wearing_id"]])
    df = pd.DataFrame(rows, columns=["person_id", "in_count", "out_count", "is_wearing_id"])
    df.to_csv(CSV_PATH, index=False)
    print("📁 CSV updated")

def detect_id_on_person(frame, box):
    x1, y1, x2, y2 = map(int, box)
    person_crop = frame[y1:y2, x1:x2]
    if person_crop.size == 0:
        return False
    results = id_model(person_crop, conf=0.5, verbose=False)[0]
    return len(results.boxes) > 0

# ------------------ MAIN LOOP ------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = person_model(frame, conf=0.5, verbose=False)[0]
    detections = []

    for box in results.boxes:
        if int(box.cls[0]) == 0:  # person
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            detections.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)
    current_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())
        x1, y1, x2, y2 = l, t, l + w, t + h

        if track_id not in trackid_to_personid:
            trackid_to_personid[track_id] = next_person_id
            people_data[next_person_id] = {"in_count": 1, "out_count": 0, "is_wearing_id": False}
            print(f"➡️ Person {next_person_id} ENTERED")
            next_person_id += 1

        person_id = trackid_to_personid[track_id]
        current_ids.add(person_id)
        last_seen[person_id] = frame_count

        wearing_id = detect_id_on_person(frame, (x1, y1, x2, y2))
        if wearing_id:
            people_data[person_id]["is_wearing_id"] = True

        color = (0, 255, 0) if wearing_id else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label1 = f"Person {person_id}"
        label2 = f"IN:{people_data[person_id]['in_count']} OUT:{people_data[person_id]['out_count']}"
        label3 = f"ID Card: {'Yes' if wearing_id else 'No'}"

        cv2.putText(frame, label1, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, label2, (x1 + 5, y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, label3, (x2 - 140, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for pid in list(last_seen.keys()):
        if pid not in current_ids and frame_count - last_seen[pid] > FRAME_LOST_THRESHOLD:
            people_data[pid]["out_count"] += 1
            print(f"⬅️ Person {pid} EXITED | OUT count = {people_data[pid]['out_count']}")
            save_to_csv(people_data)
            del last_seen[pid]

    cv2.imshow("Office Entry-Exit + ID Card Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
save_to_csv(people_data)
print("✅ Final CSV saved to:", CSV_PATH)
