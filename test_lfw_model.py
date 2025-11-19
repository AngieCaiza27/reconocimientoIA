# test_lfw_model.py
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import cv2
import os
from predecir_lfw import predecir_lfw




# Crear carpeta de pruebas si no existe
os.makedirs("pruebas", exist_ok=True)

print("Cargando dataset LFW...")
lfw = fetch_lfw_people(min_faces_per_person=50, resize=0.5)

# Tomamos una imagen del dataset (puedes cambiar el índice)
index = 0
img_raw = lfw.images[index]
label_real = lfw.target_names[lfw.target[index]]

print("\nNombre real:", label_real)

# Guardar la imagen en gris en la carpeta pruebas
imagen_path = "pruebas/test_lfw.jpg"
plt.imsave(imagen_path, img_raw, cmap="gray")

print("Imagen guardada en:", imagen_path)

# PROBAR el modelo con esa imagen
print("\nProbando modelo...")
prediccion = predecir_lfw(imagen_path)

print("\nRESULTADO DEL MODELO:", prediccion)

# Mostrar la imagen
plt.imshow(img_raw, cmap="gray")
plt.title(f"Verdadero: {label_real}\nPredicción: {prediccion}")
plt.axis("off")
plt.show()
