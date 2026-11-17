# predict_hibrido.py
import cv2
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace

import json
import os

# 🔥 Umbral recomendado (FaceNet512)
UMBRAL_DESCONOCIDO = 12.0


# ============================================================
# 📌 Leer los embeddings guardados
# ============================================================
def cargar_embeddings():
    file = "embeddings_hibrido.json"
    if not os.path.exists(file):
        print("⚠ No existen embeddings aún.")
        return []
    with open(file, "r") as f:
        return json.load(f)


# ============================================================
# 📌 Obtener embedding del rostro
# ============================================================
def obtener_embedding(path_img):
    emb_info = DeepFace.represent(
        img_path=path_img,
        model_name="Facenet512",
        detector_backend="retinaface",
        enforce_detection=False
    )[0]

    return np.array(emb_info["embedding"])


# ============================================================
# 📌 Detectar rostro con RetinaFace
# ============================================================
def detectar_rostro(img):
    try:
        det = RetinaFace.detect_faces(img)
    except:
        return None, None

    if not isinstance(det, dict) or len(det) == 0:
        return None, None

    key = list(det.keys())[0]
    x1, y1, x2, y2 = det[key]["facial_area"]

    rostro = img[y1:y2, x1:x2]
    return rostro, (x1, y1, x2, y2)


# ============================================================
# 📌 Predicción híbrida
# ============================================================
def predecir_rostro_hibrido(path_img):
    embeddings = cargar_embeddings()
    if not embeddings:
        return "Sin datos", None

    img = cv2.imread(path_img)
    rostro, box = detectar_rostro(img)

    if rostro is None:
        return "No detectado", None

    emb_new = obtener_embedding(path_img)

    mejor_dist = 9999
    mejor_nombre = "Desconocido"

    for item in embeddings:
        emb_reg = np.array(item["embedding"])
        dist = np.linalg.norm(emb_new - emb_reg)

        if dist < mejor_dist:
            mejor_dist = dist
            mejor_nombre = item["name"]

    # Aplicar umbral
    if mejor_dist > UMBRAL_DESCONOCIDO:
        mejor_nombre = "Desconocido"

    return mejor_nombre, rostro
