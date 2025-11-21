# webcam_realtime.py
import cv2
import numpy as np
import joblib
from deepface import DeepFace
from retinaface import RetinaFace

clf = joblib.load("modelo/clasificador.pkl")
encoder = joblib.load("modelo/encoder.pkl")

def predecir_frame(frame):

    detections = RetinaFace.detect_faces(frame)

    if not isinstance(detections, dict):
        return frame

    for _, det in detections.items():
        x1, y1, x2, y2 = det["facial_area"]
        rostro = frame[y1:y2, x1:x2]

        if rostro.size == 0:
            continue

        try:
            # ⛔ DeepFace espera array en input_img, NO en img_path
            emb_info = DeepFace.represent(
                input_img=rostro,
                model_name="Facenet512",
                detector_backend="skip",
                enforce_detection=False
            )[0]

            emb = np.array(emb_info["embedding"]).reshape(1, -1)

            pred = clf.predict(emb)
            nombre = encoder.inverse_transform(pred)[0]
        except Exception:
            nombre = "Desconocido"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, nombre, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return

    print("🎥 Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_proc = predecir_frame(frame)
        cv2.imshow("Reconocimiento en tiempo real", frame_proc)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
