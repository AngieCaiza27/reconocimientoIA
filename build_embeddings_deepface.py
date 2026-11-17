# build_embeddings_deepface.py
import os
import json
from deepface import DeepFace

DATASET_DIR = r"C:\Users\Betsabe\Desktop\IA\ReconocimeintoIA\archive\lfw-deepfunneled\lfw-deepfunneled"
OUTPUT_EMB = "modelo/embeddings.json"

def build_embeddings():
    print("🔍 Buscando imágenes en el dataset...")
    embeddings = []

    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)

        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            if not img_name.lower().endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(person_path, img_name)

            try:
                print(f"Procesando: {img_path}")
                result = DeepFace.represent(
                    img_path,
                    model_name="Facenet512",
                    detector_backend="retinaface",  # preciso y estable
                    enforce_detection=False
                )

                embeddings.append({
                    "name": person_name,
                    "embedding": result[0]["embedding"]
                })

            except Exception as e:
                print(f"❌ Error con {img_path}: {e}")

    print("💾 Guardando embeddings...")
    os.makedirs("modelo", exist_ok=True)

    with open(OUTPUT_EMB, "w") as f:
        json.dump(embeddings, f)

    print("✅ Embeddings creados correctamente!")


if __name__ == "__main__":
    build_embeddings()
