# rtsp_stream.py

import cv2
import threading

class RTSPVideoStream:
    def __init__(self, rtsp_url, width=640, height=480):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height

        # Use FFMPEG backend for stability
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not self.cap.isOpened():
            raise Exception("Error: Could not open RTSP stream")

        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if not grabbed:
                continue

            frame = cv2.resize(frame, (self.width, self.height))

            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()
