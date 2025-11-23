import os
import json
import numpy as np
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

EMBEDDINGS_FILE = "embeddings.json"
MODEL_DIR = "modelo"
CLF_PATH = os.path.join(MODEL_DIR, "clasificador.pkl")
ENC_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

def cargar_embeddings():
    """
    Carga SOLO embeddings.json (nuevos registros)
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        return []
    with open(EMBEDDINGS_FILE, "r") as f:
        return json.load(f)

def guardar_embeddings(data):
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(data, f, indent=4)

def agregar_rostro_hibrido(image_path, nombre):
    print(f"🔍 Generando embedding para: {nombre}")
    
    backends = ['retinaface', 'mtcnn', 'opencv', 'ssd']
    
    for backend in backends:
        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet512",
                detector_backend=backend,
                enforce_detection=False
            )
            
            if isinstance(result, list) and len(result) > 0:
                emb_info = result[0]
                vector = emb_info["embedding"]
                
                data = cargar_embeddings()
                data.append({
                    "name": nombre,
                    "embedding": vector
                })
                guardar_embeddings(data)
                print(f"✅ Embedding agregado para {nombre} (detector: {backend})")
                return True
                
        except Exception as e:
            print(f"⚠️ Error con {backend}: {e}")
            continue
    
    print(f"❌ No se pudo generar embedding para {nombre}")
    return False

def reentrenar_hibrido():
    """
    Reentrena el SVM usando SOLO embeddings.json (nuevos registros)
    NO usa modelo/embeddings.json para evitar reentrenar con muchos datos
    """
    # Cargar SOLO embeddings.json
    data = cargar_embeddings()
    
    if len(data) < 2:
        print("⚠ Se necesitan al menos 2 personas en embeddings.json para entrenar SVM.")
        print("💡 Nota: El modelo preentrenado de modelo/embeddings.json sigue disponible.")
        return False
    
    print(f"📚 Reentrenando SVM con {len(data)} embeddings de embeddings.json...")
    
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
    
    personas = len(set(nombres))
    print(f"🎉 Modelo SVM reentrenado con {personas} personas nuevas.")
    print(f"💡 El modelo seguirá usando modelo/embeddings.json para distancia coseno.")
    return True