import cv2
import os

video_path = "dataset/videos/Resham.mp4"
output_dir = "known_faces/Resham"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
save_every = 5 # save every 5th frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % save_every == 0:
        cv2.imwrite(f"{output_dir}/frame_{frame_count}.jpg", frame)

    frame_count += 1

cap.release()
print("Frames extracted successfully!")
