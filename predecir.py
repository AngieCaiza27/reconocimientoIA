# predecir.py con Mediapipe (detector de rostros real)
import joblib
import cv2
import mediapipe as mp
import numpy as np

# Cargar modelo entrenado
clf = joblib.load("modelo/clasificador.pkl")
pca = joblib.load("modelo/pca.pkl")
scaler = joblib.load("modelo/scaler.pkl")
names = joblib.load("modelo/nombres.pkl")

mp_face = mp.solutions.face_detection


def predecir_rostro(imagen_path, umbral_prob=0.7, umbral_margin=0.5, model_selection=1, min_conf=0.6):
    img = cv2.imread(imagen_path)
    if img is None:
        return "Error: Imagen no encontrada.", None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detector de rostros mediapipe
    with mp_face.FaceDetection(model_selection=model_selection, min_detection_confidence=min_conf) as face_detection:
        results = face_detection.process(img_rgb)

        if not results.detections:
            return "No se detecta rostro humano.", img, None

        # Elegir la detección con mayor score
        detection = max(results.detections, key=lambda d: d.score[0] if d.score else 0.0)
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = img.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        ww = int(bbox.width * w)
        hh = int(bbox.height * h)

        # Añadir margen alrededor del rostro (p. ej. 15%) y asegurar límites de imagen
        m = 0.15
        x = int(max(0, x - m * ww))
        y = int(max(0, y - m * hh))
        ww = int(min(w - x, ww * (1 + 2 * m)))
        hh = int(min(h - y, hh * (1 + 2 * m)))

        # Asegurar recorte válido
        if ww <= 0 or hh <= 0:
            return "Rostro no válido.", img, None
        rostro = img[y:y+hh, x:x+ww]

        if rostro.size == 0:
            return "Rostro no válido.", img, None

        # Preprocesado: gris, (opcional) normalización local, resize a LFW
        rostro_gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            rostro_gray = clahe.apply(rostro_gray)
        except Exception:
            pass
        # LFW usa (alto=62, ancho=47), OpenCV recibe (width, height)
        rostro_resized = cv2.resize(rostro_gray, (47, 62))

        # Flatten + preprocesado
        rostro_flat = rostro_resized.flatten()
        x_scaled = scaler.transform([rostro_flat])
        x_pca = pca.transform(x_scaled)

        # Predicción con umbral de "Desconocido"
        etiqueta = None
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(x_pca)[0]
            idx = int(np.argmax(probs))
            if probs[idx] >= umbral_prob:
                etiqueta = names[idx]
            else:
                etiqueta = "Desconocido"
        else:
            # Fallback sin probabilidades: usar margen de decisión
            scores = clf.decision_function(x_pca)
            if scores.ndim == 1:
                max_score = float(scores)
                idx = int(max_score >= 0)
            else:
                max_score = float(np.max(scores))
                idx = int(np.argmax(scores))
            if max_score >= umbral_margin:
                etiqueta = names[idx]
            else:
                etiqueta = "Desconocido"

        # Dibujar caja en imagen
        img_box = img.copy()
        cv2.rectangle(img_box, (x, y), (x + ww, y + hh), (0, 255, 0), 2)

        return etiqueta, img_box, rostro_resized
