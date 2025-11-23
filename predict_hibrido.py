import cv2
import numpy as np
from deepface import DeepFace
import json
import os
import joblib

UMBRAL_DESCONOCIDO = 0.55

def cargar_embeddings():
    if not os.path.exists("embeddings.json"):
        return []
    with open("embeddings.json", "r") as f:
        return json.load(f)

def obtener_embedding(path_img):
    emb_info = DeepFace.represent(
        img_path=path_img,
        model_name="Facenet512",
        detector_backend="retinaface",
        enforce_detection=False
    )[0]
    return np.array(emb_info["embedding"])

# ============================================================
# SOLUCIÓN: Usar DeepFace para detectar Y extraer rostro
# ============================================================
def detectar_rostro(path_img):
    img = cv2.imread(path_img)
    if img is None:
        print("No se pudo cargar la imagen")
        return None, None
    
    try:
        # Usar DeepFace para extraer rostros (más robusto)
        face_objs = DeepFace.extract_faces(
            img_path=path_img,
            detector_backend='retinaface',
            enforce_detection=False,
            align=True
        )
        
        if not face_objs or len(face_objs) == 0:
            print("No se detectó ningún rostro")
            return None, None
        
        # Obtener el primer rostro detectado
        face_obj = face_objs[0]
        facial_area = face_obj['facial_area']
        
        x = facial_area['x']
        y = facial_area['y']
        w = facial_area['w']
        h = facial_area['h']
        
        # Extraer rostro de la imagen original
        rostro = img[y:y+h, x:x+w]
        
        # print(f"Rostro detectado en: x={x}, y={y}, w={w}, h={h}")
        return rostro, img
        
    except Exception as e:
        print(f"Error en detección: {e}")
        return None, None

def distancia_coseno(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def predecir_rostro_hibrido(path_img):
    rostro, original = detectar_rostro(path_img)
    if rostro is None:
        return "No detectado", None

    emb = obtener_embedding(path_img)
    embeddings = cargar_embeddings()

    # 1️⃣ CLASIFICACIÓN SUPERVISADA (SVM)
    if os.path.exists("modelo/clasificador.pkl"):
        clf = joblib.load("modelo/clasificador.pkl")
        encoder = joblib.load("modelo/encoder.pkl")

        probs = clf.predict_proba([emb])[0]
        pred_class = clf.predict([emb])[0]

        nombre_svm = encoder.inverse_transform([pred_class])[0]
        confianza = max(probs)

        if confianza >= 0.60:
            return nombre_svm, rostro

    # 2️⃣ DISTANCIA COSENO
    if len(embeddings) == 0:
        return "Sin base de datos", rostro
    
    distancias = []
    for d in embeddings:
        d_emb = np.array(d["embedding"])
        dist = distancia_coseno(emb, d_emb)
        distancias.append((dist, d["name"]))

    dist, nombre = min(distancias, key=lambda x: x[0])

    if dist > UMBRAL_DESCONOCIDO:
        return "Desconocido", rostro

    return nombre, rostro