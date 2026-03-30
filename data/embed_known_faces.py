
import os
import numpy as np
from deepface import DeepFace

KNOWN_FACES_DIR = "data/known_faces"
EMBEDDINGS_FILE = os.path.join(KNOWN_FACES_DIR, "known_faces_embeddings.npz")
MODEL_NAME = "Facenet"

# Ensure the directory exists
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

embeddings = []
labels = []

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        try:
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                enforce_detection=False
            )[0]["embedding"]
            embeddings.append(embedding)
            labels.append(person_name)
            print(f"Embedded: {img_path}")
        except Exception as e:
            print(f"Failed: {img_path} ({e})")

embeddings = np.array(embeddings)
labels = np.array(labels)
np.savez_compressed(EMBEDDINGS_FILE, embeddings=embeddings, labels=labels)
print(f"Saved {len(embeddings)} embeddings to {EMBEDDINGS_FILE}")