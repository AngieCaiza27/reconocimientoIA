# predict_deepface.py
import cv2
import numpy as np
import joblib
from deepface import DeepFace
from retinaface import RetinaFace

# === CARGA DE MODELOS ===
clf = joblib.load("modelo/clasificador.pkl")
encoder = joblib.load("modelo/encoder.pkl")

def predecir_rostro(imagen_path):
    # Leer imagen
    img = cv2.imread(imagen_path)

    if img is None:
        return "ERROR: Imagen no encontrada.", None

    # ----------------------
    # 1. DETECTAR ROSTRO
    # ----------------------
    try:
        detections = RetinaFace.detect_faces(imagen_path)
    except Exception as e:
        return f"Error al detectar rostro: {e}", None

    if not isinstance(detections, dict):
        return "No se detecta rostro en la imagen.", None

    # Tomamos el primer rostro del diccionario
    face_key = list(detections.keys())[0]
    face_info = detections[face_key]

    # RetinaFace devuelve "facial_area" como lista [x1, y1, x2, y2]
    x1, y1, x2, y2 = face_info["facial_area"]

    # recorte
    rostro = img[y1:y2, x1:x2]

    if rostro.size == 0:
        return "Rostro vacío o inválido.", None

    # ----------------------
    # 2. OBTENER EMBEDDING con FaceNet512
    # ----------------------
    try:
        embedding_info = DeepFace.represent(
            img_path=imagen_path,
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=False
        )[0]

        embedding = np.array(embedding_info["embedding"]).reshape(1, -1)
    except Exception as e:
        return f"Error al generar embedding: {e}", None

    # ----------------------
    # 3. PREDECIR
    # ----------------------
    pred = clf.predict(embedding)
    nombre = encoder.inverse_transform(pred)[0]

    # Dibujar caja
    img_box = img.copy()
    cv2.rectangle(img_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return nombre, img_box
