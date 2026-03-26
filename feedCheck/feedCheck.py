import cv2
import os
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from feedCheck.rtsp_stream import RTSPVideoStream   # YOUR custom RTSP class

# Config 
PERSON_MODEL_PATH = "models/yolov8n.pt"
ID_MODEL_PATH = "models/runs/detect/retrain_v2/weights/best.pt"
KNOWN_FACES_DIR = "data/known_faces"
CONF_THRESHOLD = 0.6

rtsp_url='rtsp://admin:Amicus%402026@192.168.2.99:554/live'

# Load models
person_model = YOLO(PERSON_MODEL_PATH)
id_model = YOLO(ID_MODEL_PATH)

# Start RTSP stream
cap = RTSPVideoStream(rtsp_url).start()

print("Press Q to exit")

def enhance_frame(frame):
    # Convert to float
    frame_float = frame.astype('float32') / 255.0

    # Increase contrast slightly
    alpha = 1.2   # contrast
    beta = 5      # brightness
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Sharpen using kernel
    kernel = [[0, -1, 0],
              [-1, 5, -1],
              [0, -1, 0]]

    kernel = np.array(kernel)
    sharpened = cv2.filter2D(frame, -1, kernel)

    return sharpened

# Face recognition function
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


# Main loop
while True:
    frame = cap.read()

    if frame is None:
        continue

    # Run detection models
    person_results = person_model(frame, conf=CONF_THRESHOLD)
    id_results = id_model(frame, conf=CONF_THRESHOLD)

    persons = person_results[0].boxes
    id_cards = id_results[0].boxes

    # Loop through each detected person
    for p in persons:
        cls = int(p.cls[0])
        conf = float(p.conf[0])

        # COCO class 0 = person
        if cls == 0 and conf > CONF_THRESHOLD:

            x1, y1, x2, y2 = map(int, p.xyxy[0])
            person_has_id = False

            # Check if any ID card box is inside person box
            for i in id_cards:
                ix1, iy1, ix2, iy2 = map(int, i.xyxy[0])

                cx = (ix1 + ix2) // 2
                cy = (iy1 + iy2) // 2

                if x1 < cx < x2 and y1 < cy < y2:
                    person_has_id = True
                    break

            # Face recognition 
            # Crop upper half of person box (face approx area)
            face_crop = frame[y1:y1 + (y2 - y1)//2, x1:x2]

            if face_crop.size != 0:
                person_name = recognize_face(face_crop)
            else:
                person_name = "Unknown"

            # Final label
            if person_has_id:
                color = (0, 255, 0)
                id_status = "ID Present"
            else:
                color = (0, 0, 255)
                id_status = "No ID"

            label = f"{person_name} | {id_status}"

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2)

    cv2.imshow("RTSP Human + ID + Face Recognition", frame)
    
    # added to enhnce resolution 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Cleanup
cap.stop()
cv2.destroyAllWindows()
