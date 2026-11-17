import cv2
import os
from retinaface import RetinaFace

from model_utils import agregar_rostro_hibrido, reentrenar_hibrido


TEMP_FILE = "captura_temp.jpg"


def capturar_rostro():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara.")
        return None

    print("📸 Coloca tu rostro frente a la cámara. Presiona 'c' para capturar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Registrar rostro", frame)

        key = cv2.waitKey(1) & 0xFF

        # Capturar rostro
        if key == ord('c'):
            try:
                detections = RetinaFace.detect_faces(frame)
            except:
                print("❌ Error al detectar rostro.")
                continue

            if not isinstance(detections, dict) or len(detections) == 0:
                print("⚠ No se detectó rostro, intenta otra vez.")
                continue

            key_face = next(iter(detections))
            x1, y1, x2, y2 = detections[key_face]["facial_area"]

            rostro = frame[y1:y2, x1:x2]

            cv2.imwrite(TEMP_FILE, rostro)
            print("✨ Rostro recortado correctamente.")
            break

        # Cancelar
        if key == ord('q'):
            print("❌ Cancelado por el usuario.")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return TEMP_FILE


def main():
    path = capturar_rostro()
    if not path:
        return

    nombre = input("👤 Ingresa tu nombre: ").strip()
    if not nombre:
        print("❌ Nombre inválido.")
        return

    # Registrar rostro
    try:
        agregar_rostro_hibrido(path, nombre)
        print(f"✅ Rostro '{nombre}' añadido correctamente.")

        print("🔄 Reentrenando modelo híbrido...")
        ok = reentrenar_hibrido()

        if ok:
            print("🎉 Modelo actualizado y listo para usar.")
        else:
            print("⚠ Se registró el rostro, pero el modelo no se pudo reentrenar.")

    except Exception as e:
        print(f"❌ Error al registrar: {e}")


if __name__ == "__main__":
    main()
