
import cv2
import os
import re
import time
import csv
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import easyocr
from fuzzywuzzy import fuzz
 
# Configuration parameters
CONF_THRESHOLD = 0.5
KNOWN_FACES_DIR = "data/known_faces"
EMBEDDINGS_FILE = "data/known_faces/known_faces_embeddings.npz"
EMBEDDING_MODEL = "Facenet"
RESET_TIME = 3  # seconds
CSV_FILE = "logs/attendance_log.csv"
 
# Initialize CSV fileif it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "ID_Detected"])
 
# Track logged persons
logged_persons = set()
 
# Load models
person_model = YOLO("models/yolov8n.pt")
id_model = YOLO("models/runs/detect/retrain_v2/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)
 
# Camera setup
cap = cv2.VideoCapture(0)
 
# State tracking for each person box
person_state = {}
last_seen_id = {}
 

# Load precomputed embeddings
def load_known_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Embeddings file {EMBEDDINGS_FILE} not found. Run embed_known_faces.py first.")
        return None, None
    data = np.load(EMBEDDINGS_FILE)
    return data['embeddings'], data['labels']

known_embeddings, known_labels = load_known_embeddings()

# Fast face recognition using precomputed embeddings
def recognize_face(face_img):
    global known_embeddings, known_labels
    if known_embeddings is None or known_labels is None:
        return "Unknown"
    try:
        # Compute embedding for the input face
        embedding = DeepFace.represent(
            img_path=face_img,
            model_name=EMBEDDING_MODEL,
            enforce_detection=False
        )[0]["embedding"]
        # Compute cosine similarity
        emb_array = np.array(known_embeddings)
        embedding = np.array(embedding)
        # Normalize
        emb_array = emb_array / np.linalg.norm(emb_array, axis=1, keepdims=True)
        embedding = embedding / np.linalg.norm(embedding)
        similarities = np.dot(emb_array, embedding)
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        # Threshold for recognition (tune as needed)
        if best_score > 0.6:
            return known_labels[best_idx]
    except Exception as e:
        print(f"Face recognition error: {e}")
    return "Unknown"
 
# OCR function to extract name from ID card
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
 
# Fuzzy matching of names
def match_names(face_name, id_name):
 
    if not id_name or not face_name or face_name == "Unknown":
        return False
 
    score_full = fuzz.ratio(face_name.lower(), id_name.lower())
    score_partial = fuzz.partial_ratio(face_name.lower(), id_name.lower())
 
    final_score = max(score_full, score_partial)
 
    print(f" MATCH → Full: {score_full}, Partial: {score_partial}, Final: {final_score}")
 
    return final_score > 75
 
# CSV Logging
def log_to_csv(name, id_detected):
 
    # Expect id_detected to be a tuple: (id_detected, id_authenticated)
    # id_authenticated: 'YES' if OCR name and face name match, else 'NO'
    if name in logged_persons:
        return
 
    # Support backward compatibility if id_detected is not a tuple
    if isinstance(id_detected, tuple):
        id_detected_val, id_authenticated = id_detected
    else:
        id_detected_val = id_detected
        id_authenticated = "NO"
 
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, id_detected_val, id_authenticated])
 
    logged_persons.add(name)
 
# Main loop
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
 
        if box_key not in person_state:
            person_state[box_key] = {
                "status": "NO_ID",
                "name": "",
                "locked": False
            }
 
        state = person_state[box_key]
 
        # Check ID
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
 
        # Reset logic
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
 
        # Main logic
        if has_id:
 
            if state["locked"]:
                label = state["name"] + " +"
                color = (0, 255, 0)
 
            else:
                state["status"] = "ID_DETECTED"
 
                face_crop = frame[y1:y1+(y2-y1)//2, x1:x2]
                face_name = recognize_face(face_crop)
 
                id_name = extract_name(id_crop) if id_crop is not None else ""
 
                if id_name and face_name != "Unknown":
                    # Only log when OCR authentication happens
                    if match_names(face_name, id_name):
                        state["status"] = "AUTHENTICATED"
                        state["name"] = face_name
                        state["locked"] = True
 
                        label = face_name + " +"
                        color = (0, 255, 0)
 
                        log_to_csv(face_name, ("YES", "YES"))
                    else:
                        state["status"] = "FAILED"
                        label = face_name + " -"
                        color = (255, 0, 255)
 
                        log_to_csv(face_name, ("YES", "NO"))
                else:
                    label = face_name
                    color = (0, 255, 255)
                    # Do not log if OCR authentication did not happen
                    pass
 
        else:
            label = "NO ID"
            color = (0, 0, 255)
            # Do not log if no ID
            pass
 
        # Draw bounding box and label
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
 