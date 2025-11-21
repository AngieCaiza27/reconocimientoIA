# model_utils.py
import os
import json
import joblib
import numpy as np
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

EMB_FILE = "modelo/embeddings.json"
CLF_FILE = "modelo/clasificador.pkl"
ENC_FILE = "modelo/encoder.pkl"

def cargar_embeddings():
    if not os.path.exists(EMB_FILE):
        return []
    with open(EMB_FILE, "r") as f:
        return json.load(f)

def guardar_embeddings(data):
    os.makedirs("modelo", exist_ok=True)
    with open(EMB_FILE, "w") as f:
        json.dump(data, f)

def agregar_rostro(imagen_path, nombre_persona):
    """Genera embedding con FaceNet512 y lo añade al JSON."""
    print(f"Generando embedding para {nombre_persona}...")
    emb_info = DeepFace.represent(
        img_path=imagen_path,
        model_name="Facenet512",
        detector_backend="retinaface",
        enforce_detection=False
    )[0]

    emb = emb_info["embedding"]

    data = cargar_embeddings()
    data.append({
        "name": nombre_persona,
        "embedding": emb
    })
    guardar_embeddings(data)
    print("✅ Rostro añadido a embeddings.json")

def reentrenar_clasificador():
    """Reentrena el SVM usando todos los embeddings."""
    data = cargar_embeddings()
    if not data:
        print("❌ No hay embeddings para entrenar.")
        return False

    X = [d["embedding"] for d in data]
    y_names = [d["name"] for d in data]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_names)

    clf = SVC(kernel="linear", probability=True)
    clf.fit(X, y)

    os.makedirs("modelo", exist_ok=True)
    joblib.dump(clf, CLF_FILE)
    joblib.dump(encoder, ENC_FILE)

    print("🎯 Clasificador reentrenado correctamente.")
    print(f"Personas: {len(set(y_names))} | Embeddings: {len(X)}")
    return True

