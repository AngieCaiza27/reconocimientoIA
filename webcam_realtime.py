import cv2
from predict_hibrido import predecir_rostro_hibrido
from retinaface import RetinaFace


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara.")
        return

    print("🎥 Reconocimiento activo — Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar rostro directamente del frame
        try:
            faces = RetinaFace.detect_faces(frame)
        except:
            faces = None

        nombre = "No detectado"

        if isinstance(faces, dict) and len(faces) > 0:
            key = next(iter(faces))
            x1, y1, x2, y2 = faces[key]["facial_area"]

            # Guardar recorte temporalmente
            rostro = frame[y1:y2, x1:x2]
            temp_path = "temp_webcam_face.jpg"
            cv2.imwrite(temp_path, rostro)

            # Predicción híbrida (SVM + Umbral)
            nombre_pred, _ = predecir_rostro_hibrido(temp_path)
            nombre = nombre_pred

            # Dibujar cuadro
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, nombre, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Reconocimiento en tiempo real", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
