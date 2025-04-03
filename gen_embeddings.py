# gen_embeddings.py
import os
import pickle
from deepface import DeepFace

DATASET_PATH = "dataset"
X = []
y = []

for person in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_dir): continue
    for img_file in os.listdir(person_dir):
        path = os.path.join(person_dir, img_file)
        try:
            emb = DeepFace.represent(img_path=path, model_name="ArcFace", enforce_detection=False)[0]['embedding']
            X.append(emb)
            y.append(person)
            print(f"✅ {person} - {img_file}")
        except Exception as e:
            print(f"❌ {path}: {e}")

# Save to file
with open("embeddings.pkl", "wb") as f:
    pickle.dump((X, y), f)

print("✅ Đã tạo embeddings.pkl")
