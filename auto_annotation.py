from ultralytics import YOLO
import os
import cv2

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

source_folder = "raw_frames"
output_images = "dataset/train/images"
output_labels = "dataset/train/labels"

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

for img_name in os.listdir(source_folder):
    img_path = os.path.join(source_folder, img_name)

    results = model(img_path, conf=0.6)  # increase confidence threshold

    # Save image to dataset folder
    img = cv2.imread(img_path)
    cv2.imwrite(os.path.join(output_images, img_name), img)

    # Create label file
    label_path = os.path.join(output_labels, img_name.replace(".jpg", ".txt"))

    with open(label_path, "w") as f:
        for r in results:
            boxes = r.boxes.xywhn  # normalized xywh
            classes = r.boxes.cls

            for box, cls in zip(boxes, classes):
                x, y, w, h = box.tolist()
                f.write(f"{int(cls)} {x} {y} {w} {h}\n")

print("Auto annotation complete ✅")
