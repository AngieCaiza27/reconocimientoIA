# interfaz.py
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from predict_deepface import predecir_rostro_deepface
import cv2

def seleccionar_imagen():
    path = filedialog.askopenfilename(filetypes=[("Imagenes", "*.jpg *.png *.jpeg")])
    if not path:
        return

    resultado, _ = predecir_rostro_deepface(path)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))

    img_pil = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(img_pil)

    label_imagen.config(image=img_tk)
    label_imagen.image = img_tk

    label_resultado.config(text="Predicción: " + resultado)


root = tk.Tk()
root.title("Reconocimiento Facial DeepFace")
root.geometry("400x550")

btn = tk.Button(root, text="Seleccionar Imagen", command=seleccionar_imagen, font=("Arial", 14))
btn.pack(pady=10)

label_imagen = tk.Label(root)
label_imagen.pack(pady=10)

label_resultado = tk.Label(root, text="Predicción:", font=("Arial", 16))
label_resultado.pack(pady=10)

root.mainloop()
