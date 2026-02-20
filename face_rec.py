import cv2
import os
from ultralytics import YOLO
from deepface import DeepFace

# ---------------- CONFIG ----------------
PERSON_MODEL_PATH = "yolov8n.pt"
ID_MODEL_PATH = "runs/detect/retrain_v2/weights/best.pt"
KNOWN_FACES_DIR = "known_faces"
CONF_THRESHOLD = 0.5

# ---------------- LOAD MODELS ----------------
person_model = YOLO(PERSON_MODEL_PATH)
id_model = YOLO(ID_MODEL_PATH)

# ---------------- FACE RECOGNITION FUNCTION ----------------
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

# ---------------- START CAMERA ----------------
cap = cv2.VideoCapture(0)

print("Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run models
    person_results = person_model(frame, conf=CONF_THRESHOLD)
    id_results = id_model(frame, conf=CONF_THRESHOLD)

    persons = person_results[0].boxes
    id_cards = id_results[0].boxes

    #  LOOP THROUGH EVERY PERSON
    for p in persons:
        cls = int(p.cls[0])
        conf = float(p.conf[0])

        if cls == 0 and conf > CONF_THRESHOLD:

            x1, y1, x2, y2 = map(int, p.xyxy[0])

            # Draw person box (default red)
            color = (0, 0, 255)
            person_has_id = False

            #  Check if any ID card is inside this person
            for i in id_cards:
                ix1, iy1, ix2, iy2 = map(int, i.xyxy[0])

                # ID center point
                cx = (ix1 + ix2) // 2
                cy = (iy1 + iy2) // 2

                if x1 < cx < x2 and y1 < cy < y2:
                    person_has_id = True
                    break

            # Face Recognition per person
            face_crop = frame[y1:y1 + (y2-y1)//2, x1:x2]

            if face_crop.size != 0:
                person_name = recognize_face(face_crop)
            else:
                person_name = "Unknown"

            # Final color + label
            if person_has_id:
                color = (0, 255, 0)
                id_status = "ID Present"
            else:
                color = (0, 0, 255)
                id_status = "No ID"

            label = f"{person_name} | {id_status}"

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

    cv2.imshow("Human + ID + Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
