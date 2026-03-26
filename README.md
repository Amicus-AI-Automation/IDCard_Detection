# ID Card Detection & Attendance System

This project is an AI-powered system for detecting ID cards and recognizing faces from video streams (webcam or RTSP), logging attendance automatically. It uses YOLO for object detection, DeepFace for face recognition, and EasyOCR for text extraction.

## Features

- **ID Card Detection:** Detects ID cards in video frames using YOLOv8.
- **Face Recognition:** Identifies known faces using DeepFace.
- **Text Extraction:** Reads text from ID cards using EasyOCR.
- **Attendance Logging:** Logs attendance to a CSV file.
- **RTSP & Webcam Support:** Works with both local webcams and RTSP streams.

## Directory Structure

- `main.py` — Main script for webcam-based detection and attendance.
- `feedCheck.py` — Script for RTSP stream-based detection.
- `rtsp_stream.py` — Custom RTSP video stream handler.
- `requirements.txt` — Python dependencies.
- `data/`
  - `known_faces/` — Subfolders for each known person, with their images.
  - `train/`, `test/`, `valid/` — YOLO dataset folders (images & labels).
- `logs/attendance_log.csv` — Attendance records.
- `models/`
  - `yolov8n.pt` — YOLOv8 base model.
  - `runs/detect/retrain_v2/weights/best.pt` — Custom-trained ID card detector.

## Setup

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
2. **Prepare data:**
   - Place known face images in `data/known_faces/<Person_Name>/`.
   - Organize YOLO datasets in `data/train`, `data/test`, `data/valid`.

3. **Run the application:**
   - For webcam:
     ```
     python main.py
     ```
   - For RTSP stream:
     ```
     python feedCheck.py
     ```

## Notes

- Attendance is logged in `logs/attendance_log.csv`.
- Models must be present in the `models/` directory.
- Update RTSP URL in `feedCheck.py` as needed.
