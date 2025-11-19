# predecir_lfw.py
import cv2
import joblib

clf = joblib.load("modelo/clasificador.pkl")
pca = joblib.load("modelo/pca.pkl")
scaler = joblib.load("modelo/scaler.pkl")
names = joblib.load("modelo/nombres.pkl")

def predecir_lfw(imagen_path):
    img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    # LFW: height=62, width=47 → OpenCV resize expects (width, height)
    img = cv2.resize(img, (47, 62))

    x = img.flatten().reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_pca = pca.transform(x_scaled)

    pred = clf.predict(x_pca)[0]
    return names[pred]
