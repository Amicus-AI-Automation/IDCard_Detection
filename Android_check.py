import cv2
import os
from ultralytics import YOLO
from deepface import DeepFace

# ---------------- CONFIG ----------------
PERSON_MODEL_PATH = "yolov8n.pt"
ID_MODEL_PATH = "runs/detect/retrain_v2/weights/best.pt"
KNOWN_FACES_DIR = "known_faces"
CONF_THRESHOLD = 0.6
IP_WEBCAM_URL = "http://10.172.68.67:8080/video"   # <-- CHANGE

FACE_RECOG_INTERVAL = 10  # Run DeepFace every 10 frames

# ---------------- LOAD MODELS ----------------
person_model = YOLO(PERSON_MODEL_PATH)
id_model = YOLO(ID_MODEL_PATH)

# ---------------- START IP WEBCAM ----------------
cap = cv2.VideoCapture(IP_WEBCAM_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error connecting to IP Webcam")
    exit()

print("Press Q to exit")

# ---------------- VARIABLES ----------------
frame_count = 0
recognized_cache = {}  # Cache names per box position

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
            identity_path = result[0].iloc[0]['identity']
            name = identity_path.split(os.sep)[-2]
            return name
        else:
            return "Unknown"
    except:
        return "Unknown"

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize for speed
    frame = cv2.resize(frame, (640, 480))

    frame_count += 1

    # Run YOLO
    person_results = person_model(frame, conf=CONF_THRESHOLD)
    id_results = id_model(frame, conf=CONF_THRESHOLD)

    persons = person_results[0].boxes
    id_cards = id_results[0].boxes

    for p in persons:
        cls = int(p.cls[0])
        conf = float(p.conf[0])

        if cls == 0 and conf > CONF_THRESHOLD:

            x1, y1, x2, y2 = map(int, p.xyxy[0])
            person_has_id = False

            # -------- CHECK ID INSIDE PERSON --------
            for i in id_cards:
                ix1, iy1, ix2, iy2 = map(int, i.xyxy[0])
                cx = (ix1 + ix2) // 2
                cy = (iy1 + iy2) // 2

                if x1 < cx < x2 and y1 < cy < y2:
                    person_has_id = True
                    break

            # -------- FACE RECOGNITION (Optimized) --------
            box_key = (x1//50, y1//50)  # rough tracking key

            if box_key not in recognized_cache or frame_count % FACE_RECOG_INTERVAL == 0:
                face_crop = frame[y1:y1 + (y2 - y1)//2, x1:x2]

                if face_crop.size != 0:
                    name = recognize_face(face_crop)
                else:
                    name = "Unknown"

                recognized_cache[box_key] = name

            person_name = recognized_cache.get(box_key, "Unknown")

            # -------- LABEL --------
            if person_has_id:
                color = (0, 255, 0)
                id_status = "ID Present"
            else:
                color = (0, 0, 255)
                id_status = "No ID"

            label = f"{person_name} | {id_status}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

    cv2.imshow("Optimized IP Webcam Human + ID + Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
