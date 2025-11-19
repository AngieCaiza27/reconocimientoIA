import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from predecir import predecir_rostro
import cv2

def seleccionar_imagen():
    path = filedialog.askopenfilename(
        filetypes=[("Archivos de imagen", "*.jpg *.png *.jpeg")]
    )

    if not path:
        return

    resultado, img_box, rostro = predecir_rostro(path)

    # Mostrar imagen original con recuadro
    if img_box is not None:
        img_rgb = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((320, 260))
        img_tk = ImageTk.PhotoImage(img_pil)
        label_imagen.config(image=img_tk)
        label_imagen.image = img_tk
    else:
        label_imagen.config(image="", text="No se pudo mostrar la imagen")

    # Mostrar rostro recortado
    if rostro is not None:
        rostro_pil = Image.fromarray(rostro)
        rostro_pil = rostro_pil.resize((120, 120))
        rostro_tk = ImageTk.PhotoImage(rostro_pil)
        label_rostro.config(image=rostro_tk)
        label_rostro.image = rostro_tk
    else:
        label_rostro.config(image="", text="No se detectó rostro")

    # Mostrar predicción
    label_resultado.config(text="Predicción: " + resultado)


# GUI
root = tk.Tk()
root.title("Clasificación de Rostros (LFW)")
root.geometry("500x700")

btn = tk.Button(root, text="Seleccionar Imagen", command=seleccionar_imagen, font=("Arial", 12))
btn.pack(pady=10)

label_imagen = tk.Label(root)
label_imagen.pack(pady=10)

label_rostro = tk.Label(root, text="Rostro recortado", font=("Arial", 12))
label_rostro.pack(pady=10)

label_resultado = tk.Label(root, text="Predicción: ---", font=("Arial", 18))
label_resultado.pack(pady=20)

root.mainloop()
