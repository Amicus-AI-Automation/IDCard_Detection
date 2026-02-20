# main.py

from rtsp_stream import RTSPVideoStream
from ultralytics import YOLO
import cv2

# RTSP URL
rtsp_url = "rtsp://admin:Amicus%402026@192.168.2.99:554/live"

# Load Person Detection Model (COCO pretrained)
person_model = YOLO("yolov8n.pt")

# Start RTSP Stream
stream = RTSPVideoStream(rtsp_url).start()

while True:

    # STEP 1: Read frame from CCTV
    frame = stream.read()

    if frame is None:
        continue

    # STEP 2: Detect persons in frame
    results = person_model.predict(frame, conf=0.4)

    for r in results:

        boxes = r.boxes.xyxy
        classes = r.boxes.cls
        scores = r.boxes.conf

        for box, cls, score in zip(boxes, classes, scores):

            if int(cls) == 0:   # 0 = person class in COCO dataset

                x1, y1, x2, y2 = map(int, box)

                # Draw PERSON box
                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (255, 0, 0),
                              2)

                cv2.putText(frame,
                            f"Person {score:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            2)

    # STEP 3: Display Output
    cv2.imshow("Person Detection Feed", frame)

    if cv2.waitKey(1) == 27:
        break

stream.stop()
cv2.destroyAllWindows()
