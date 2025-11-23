import cv2
import numpy as np
from deepface import DeepFace
import json
import os
import joblib

UMBRAL_DESCONOCIDO = 0.5

def cargar_todos_embeddings():
    """
    Carga embeddings de AMBOS archivos para predicción
    """
    embeddings_todos = []
    
    # 1️⃣ Cargar embeddings.json (nuevos registros)
    if os.path.exists("embeddings.json"):
        with open("embeddings.json", "r") as f:
            embeddings_nuevos = json.load(f)
            embeddings_todos.extend(embeddings_nuevos)
            print(f"✓ Cargados {len(embeddings_nuevos)} embeddings de embeddings.json")
    
    # 2️⃣ Cargar modelo/embeddings.json (modelo preentrenado)
    if os.path.exists("modelo/embeddings.json"):
        with open("modelo/embeddings.json", "r") as f:
            embeddings_modelo = json.load(f)
            embeddings_todos.extend(embeddings_modelo)
            print(f"✓ Cargados {len(embeddings_modelo)} embeddings de modelo/embeddings.json")
    
    print(f"📊 Total de embeddings disponibles: {len(embeddings_todos)}")
    return embeddings_todos

def obtener_embedding_y_rostro(path_img):
    """
    Extrae embedding Y rostro en una sola llamada
    """
    print(f"=" * 50)
    print(f"📂 Procesando: {path_img}")
    
    img = cv2.imread(path_img)
    if img is None:
        print(f"❌ No se puede leer la imagen")
        return None, None, None
    
    print(f"✓ Imagen cargada: {img.shape}")
    
    backends = ['retinaface', 'mtcnn', 'opencv', 'ssd']
    
    for backend in backends:
        try:
            print(f"🔍 Probando detector: {backend}...")
            
            result = DeepFace.represent(
                img_path=path_img,
                model_name="Facenet512",
                detector_backend=backend,
                enforce_detection=False
            )
            
            if not isinstance(result, list) or len(result) == 0:
                continue
            
            emb_info = result[0]
            
            if not isinstance(emb_info, dict):
                continue
            
            embedding = np.array(emb_info["embedding"])
            facial_area = emb_info.get("facial_area")
            
            print(f"✓ Embedding obtenido con {backend}")
            
            if facial_area:
                x = facial_area['x']
                y = facial_area['y']
                w = facial_area['w']
                h = facial_area['h']
                
                x = max(0, x)
                y = max(0, y)
                h_img, w_img = img.shape[:2]
                w = min(w, w_img - x)
                h = min(h, h_img - y)
                
                rostro = img[y:y+h, x:x+w]
                
                if rostro.size > 0:
                    print(f"✅ ÉXITO con {backend}")
                    print(f"=" * 50)
                    return embedding, rostro, img
            
            return embedding, None, img
            
        except Exception as e:
            continue
    
    print("❌ No se pudo detectar con ningún backend")
    print(f"=" * 50)
    return None, None, None

def distancia_coseno(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def predecir_rostro_hibrido(path_img):
    # Obtener embedding y rostro
    emb, rostro, original = obtener_embedding_y_rostro(path_img)
    
    if emb is None:
        return "No detectado", None
    
    # Cargar TODOS los embeddings (de ambos archivos) para predicción
    embeddings = cargar_todos_embeddings()
    
    # 1️⃣ CLASIFICACIÓN SUPERVISADA (SVM) - usa el modelo preentrenado
    if os.path.exists("modelo/clasificador.pkl"):
        try:
            clf = joblib.load("modelo/clasificador.pkl")
            encoder = joblib.load("modelo/encoder.pkl")
            
            probs = clf.predict_proba([emb])[0]
            pred_class = clf.predict([emb])[0]
            
            nombre_svm = encoder.inverse_transform([pred_class])[0]
            confianza = max(probs)
            
            print(f"🤖 SVM predice: {nombre_svm} (confianza: {confianza:.2f})")
            
            if confianza >= 0.60:
                return nombre_svm, rostro
        except Exception as e:
            print(f"⚠️ Error en SVM: {e}")
    
    # 2️⃣ DISTANCIA COSENO (usando TODOS los embeddings)
    if len(embeddings) == 0:
        return "Sin base de datos", rostro
    
    distancias = []
    for d in embeddings:
        d_emb = np.array(d["embedding"])
        dist = distancia_coseno(emb, d_emb)
        distancias.append((dist, d["name"]))
    
    # Ordenar y mostrar las 3 mejores coincidencias
    distancias_ordenadas = sorted(distancias, key=lambda x: x[0])
    print(f"\n📊 Top 3 coincidencias:")
    for i, (d, n) in enumerate(distancias_ordenadas[:3], 1):
        print(f"  {i}. {n}: {d:.4f}")
    
    dist, nombre = distancias_ordenadas[0]
    
    if dist > UMBRAL_DESCONOCIDO:
        print(f"❌ Distancia {dist:.4f} > umbral {UMBRAL_DESCONOCIDO}")
        return "Desconocido", rostro
    
    print(f"✅ Identificado: {nombre} (distancia: {dist:.4f})")
    return nombre, rostro