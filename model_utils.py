import os
import json
import numpy as np
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

# --------------------------------------------------------
# Archivos del modelo
# --------------------------------------------------------
EMBEDDINGS_FILE = "embeddings.json"
MODEL_DIR = "modelo"
CLF_PATH = os.path.join(MODEL_DIR, "clasificador.pkl")
ENC_PATH = os.path.join(MODEL_DIR, "encoder.pkl")


# ============================================================
# Cargar embeddings desde embeddings.json
# ============================================================
def cargar_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        return []

    with open(EMBEDDINGS_FILE, "r") as f:
        return json.load(f)


# ============================================================
# Guardar embeddings en archivo
# ============================================================
def guardar_embeddings(data):
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ============================================================
# Agregar nuevo rostro → genera embedding y lo guarda
# ============================================================
def agregar_rostro_hibrido(image_path, nombre):
    print(f"🔍 Generando embedding para: {nombre}")

    emb_info = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet512",
        detector_backend="retinaface",
        enforce_detection=False
    )[0]

    vector = emb_info["embedding"]

    data = cargar_embeddings()
    data.append({
        "name": nombre,
        "embedding": vector
    })

    guardar_embeddings(data)
    print(f"✅ Embedding agregado para {nombre}")


# ============================================================
# ENTRENAMIENTO SUPERVISADO SVM
# ============================================================
def reentrenar_hibrido():
    data = cargar_embeddings()

    if len(data) < 2:
        print("⚠ Se necesitan al menos 2 personas para entrenar SVM.")
        return False

    print(f"📚 Reentrenando SVM con {len(data)} embeddings...")

    # Matriz X e y
    embeddings = [np.array(d["embedding"]) for d in data]
    nombres = [d["name"] for d in data]

    # Codificación de etiquetas
    encoder = LabelEncoder()
    y = encoder.fit_transform(nombres)

    # Modelo SVM supervisado
    clf = SVC(kernel="linear", probability=True)
    clf.fit(embeddings, y)

    # Crear carpeta modelo
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Guardar modelo y encoder
    joblib.dump(clf, CLF_PATH)
    joblib.dump(encoder, ENC_PATH)

    print("🎉 Modelo SVM reentrenado exitosamente.")
    return True
