# predict_deepface.py
import cv2
import json
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace

# --------------------------------------
# CARGAR embeddings registrados
# --------------------------------------
with open("modelo/embeddings.json", "r") as f:
    data = json.load(f)

registered_names = [item["name"] for item in data]
registered_embeddings = np.array([item["embedding"] for item in data], dtype=np.float32)

# --------------------------------------
# DISTANCIA COSENO
# --------------------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


UMBRAL_DESCONOCIDO = 0.74   # Ajuste de desconcocido

def predecir_rostro(imagen_path):

    try:
        detections = RetinaFace.detect_faces(imagen_path)
    except Exception as e:
        return f"Error detectando rostro: {e}", None

    if detections is None or detections == {}:
        return "No se detecta rostro.", None

    # tomar primer rostro
    key = list(detections.keys())[0]
    x1, y1, x2, y2 = detections[key]["facial_area"]

    img = cv2.imread(imagen_path)
    rostro = img[y1:y2, x1:x2]

    if rostro.size == 0:
        return "Rostro inválido.", None

    # embedding con FaceNet512
    emb_info = DeepFace.represent(
        img_path=imagen_path,
        model_name="Facenet512",
        detector_backend="retinaface",
        enforce_detection=False
    )[0]

    emb = np.array(emb_info["embedding"])

    # calcular similitudes
    similitudes = [cosine_similarity(emb, ref) for ref in registered_embeddings]
    mejor = np.argmax(similitudes)
    mejor_sim = similitudes[mejor]

    if mejor_sim < UMBRAL_DESCONOCIDO:
        pred = "Desconocido"
    else:
        pred = registered_names[mejor]

    # dibujar caja
    img2 = img.copy()
    cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return pred, img2
