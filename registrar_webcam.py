import cv2
import json
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace

# Rutas
EMBEDDINGS_FILE = "modelo/embeddings.json"
from model_utils import reentrenar_clasificador


def registrar_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return

    print("📸 Coloca tu rostro frente a la cámara. Presiona 'c' para capturar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Registrar rostro - Presiona C", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            print("✨ Captura tomada.")
            break
        elif key == ord('q'):
            print("Cancelado.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    # ---------------------------
    # DETECTAR ROSTRO
    # ---------------------------
    detections = RetinaFace.detect_faces(frame)

    if not isinstance(detections, dict):
        print("❌ No se detectó un rostro claro. Intenta otra vez.")
        return

    key = list(detections.keys())[0]
    x1, y1, x2, y2 = detections[key]["facial_area"]

    rostro = frame[y1:y2, x1:x2]

    if rostro.size == 0:
        print("❌ Rostro inválido.")
        return

    # ---------------------------
    # GENERAR EMBEDDING
    # ---------------------------
    emb_info = DeepFace.represent(
        img_path=frame,
        model_name="Facenet512",
        detector_backend="retinaface",
        enforce_detection=False
    )[0]

    embedding = emb_info["embedding"]

    nombre = input("👤 Ingresa tu nombre: ")

    # ---------------------------
    # GUARDAR EN EMBEDDINGS.JSON
    # ---------------------------
    try:
        with open(EMBEDDINGS_FILE, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append({
        "name": nombre,
        "embedding": embedding
    })

    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Rostro '{nombre}' registrado correctamente.")

    # ---------------------------
    # REENTRENAR CLASIFICADOR
    # ---------------------------
    if reentrenar_clasificador():
        print("📚 Clasificador actualizado exitosamente.")
    else:
        print("⚠ Hubo un problema al reentrenar el modelo.")


if __name__ == "__main__":
    registrar_webcam()
