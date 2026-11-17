# 🎯 Sistema de Reconocimiento Facial Híbrido (DeepFace + SVM)

Sistema completo e interactivo de **reconocimiento y registro facial** en tiempo real desarrollado en Python. Utiliza modelos de Deep Learning de última generación (**FaceNet512 / DeepFace**) para la extracción de embeddings biométricos, combinados con algoritmos de Machine Learning (**SVM + PCA**) y validación por umbral de similitud para clasificar e identificar personas con alta precisión.

---

## 🌟 Características Principales

- 🧠 **Extracción de Embeddings Biométricos**: Generación de vectores de características faciales de 512 dimensiones usando **FaceNet512** (vía DeepFace).
- 🔍 **Detección Facial Multi-Backend**: Detección robusta tolerante a fallos con soporte dinámico para **RetinaFace**, **MTCNN**, **OpenCV (Haar/DNN)** y **SSD**.
- 🎛 **Arquitectura de Predicción Híbrida**:
  - Clasificador supervisado **SVM** (con reducción de dimensionalidad **PCA** y normalización `StandardScaler`).
  - Comparación de similitud coseno/euclídea directa con umbral para detectar caras desconocidas (`Desconocido`).
- 🖥 **Interfaz Gráfica de Usuario (GUI)**: Aplicación amigable construida en **Tkinter / Pillow** para cargar imágenes, predecir identidades y registrar nuevas personas sin reiniciar el sistema.
- 📸 **Reconocimiento y Registro por Webcam**:
  - Detección e inferencia de rostros en tiempo real desde la cámara web.
  - Asistente de registro interactivo con captura instantánea mediante webcam.
- 🔄 **Reentrenamiento y Actualización Dinámica**: Posibilidad de registrar nuevos rostros y reentrenar el clasificador en caliente.

---

## 🏗 Arquitectura del Proyecto

```text
reconocimientoIA/
│
├── modelo/                          # Modelos entrenados y transformadores
│   ├── clasificador.pkl             # Clasificador SVM entrenado
│   ├── pca.pkl                      # Transformador PCA ajustado
│   ├── scaler.pkl                   # StandardScaler ajustado
│   ├── nombres.pkl                  # Mapeo de identificadores y etiquetas de clases
│   └── metadata.json                # Metadatos del entrenamiento
│
├── gui_interfaz.py                  # Aplicación de escritorio principal (Tkinter)
├── predict_hibrido.py               # Lógica de inferencia híbrida (SVM + Similitud)
├── model_utils.py                   # Utilidades para registro de embeddings y reentrenamiento
├── registrar_webcam.py              # Script interactivo de captura y registro por webcam
├── webcam_realtime.py               # Reconocimiento facial en tiempo real por streaming de webcam
│
├── build_embeddings_deepface.py     # Extracción masiva de embeddings desde un dataset
├── train_classifier_cv.py           # Entrenamiento del modelo SVM con validación cruzada y tuning
├── train_classifier_deepface.py     # Script complementario de entrenamiento
├── train_model.py                   # Flujo básico de entrenamiento de modelo
│
├── install_dependencies.py          # Script de verificación e instalación de dependencias
├── requirements.txt                 # Lista de librerías y versiones requeridas
└── README.md                        # Documentación técnica del proyecto
```

---

## 📁 Dataset Base

El proyecto está configurado para trabajar con datasets estándar como **LFW (Labeled Faces in the Wild - DeepFunneled)**:
- 🔗 Disponible en Kaggle: [LFW Dataset - DeepFunneled](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
- Estructura esperada para entrenamiento masivo:
  ```text
  lfw-deepfunneled/
  ├── Persona_1/
  │   ├── Persona_1_0001.jpg
  │   └── Persona_1_0002.jpg
  ├── Persona_2/
  │   └── Persona_2_0001.jpg
  ```

---

## 🚀 Instalación y Requisitos

### Prerrequisitos
- **Python**: Versión recomendada `3.9` o `3.10`.
- Cámara web funcional (para las funciones de detección y registro en vivo).

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/reconocimientoIA.git
cd reconocimientoIA
```

### 2. Crear y activar un entorno virtual (recomendado)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
Puedes instalarlas directamente con `pip` o mediante el script asistente:

```bash
pip install -r requirements.txt
```
*(Opcional: ejecutar `python install_dependencies.py`)*

> [!NOTE]
> La primera vez que se ejecuten modelos de DeepFace (FaceNet512, RetinaFace, etc.), TensorFlow descargará automáticamente los pesos preentrenados a la caché local del usuario (`~/.deepface/weights/`).

---

## 💻 Guía de Uso

### 1. Interfaz Gráfica (Tkinter)
Inicia la aplicación visual completa:
```bash
python gui_interfaz.py
```
**Funcionalidades en la interfaz:**
- **Cargar imagen**: Selecciona cualquier imagen (`.jpg`, `.png`, `.bmp`) desde tu explorador de archivos.
- **Predecir rostro**: Detecta la cara, extrae el embedding y muestra el nombre predicho o si la persona es `Desconocido`.
- **Registrar imagen**: Asocia una imagen cargada con el nombre de una persona para guardarla en la base de datos de embeddings.
- **Registrar con Webcam**: Abre la cámara para capturar y registrar el rostro en caliente.
- **Webcam en tiempo real**: Abre el visor continuo con etiquetado dinámico de caras.

---

### 2. Reconocimiento Facial en Tiempo Real (Webcam)
Para lanzar directamente el feed de video continuo:
```bash
python webcam_realtime.py
```
- Presiona la tecla **`q`** para cerrar la ventana.

---

### 3. Registro Rápido desde Terminal + Webcam
Para capturar y registrar a una nueva persona directamente:
```bash
python registrar_webcam.py
```
- Presiona **`c`** para capturar el rostro encuadrado.
- Ingresa el nombre de la persona en la consola cuando se te solicite.
- El sistema guardará el embedding y reentrenará el clasificador automáticamente.

---

### 4. Entrenamiento y Construcción de Embeddings
Si deseas entrenar el sistema desde cero con tu propio dataset o con LFW:

1. **Construir embeddings del dataset:**
   ```bash
   python build_embeddings_deepface.py
   ```
2. **Entrenar el clasificador SVM y guardarlo:**
   ```bash
   python train_classifier_cv.py
   ```
   Esto generará los archivos correspondientes en la carpeta `modelo/` (`clasificador.pkl`, `pca.pkl`, `scaler.pkl`, `nombres.pkl`).

---

## 🛠 Tecnologías Utilizadas

| Categoría | Tecnologías / Librerías |
| :--- | :--- |
| **Deep Learning** | `TensorFlow`, `Keras`, `DeepFace` (FaceNet512) |
| **Visión por Computador** | `OpenCV (cv2)`, `RetinaFace`, `Pillow (PIL)` |
| **Machine Learning** | `scikit-learn` (SVM, PCA, GridSearchCV, StandardScaler), `joblib` |
| **Interfaz & Utilidades** | `Tkinter`, `NumPy`, `JSON` |

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia **MIT**. Consulta el archivo `LICENSE` para más detalles.
