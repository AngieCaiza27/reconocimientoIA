# gui_interfaz.py
import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2

from predict_deepface import predecir_rostro
from model_utils import agregar_rostro, reentrenar_clasificador

RUTA_ULTIMA_IMAGEN = None

class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificación de Rostros - DeepFace / FaceNet")
        self.root.geometry("900x650")

        self.label_imagen = tk.Label(root, text="No hay imagen cargada")
        self.label_imagen.pack(pady=10)

        self.label_pred = tk.Label(root, text="Predicción: -", font=("Arial", 16))
        self.label_pred.pack(pady=10)

        frame_botones = tk.Frame(root)
        frame_botones.pack(pady=10)

        btn_cargar = tk.Button(frame_botones, text="Cargar imagen", width=18, command=self.cargar_imagen)
        btn_cargar.grid(row=0, column=0, padx=5)

        btn_predecir = tk.Button(frame_botones, text="Predecir rostro", width=18, command=self.predecir)
        btn_predecir.grid(row=0, column=1, padx=5)

        btn_registrar = tk.Button(frame_botones, text="Registrar imagen", width=18, command=self.registrar_rostro)
        btn_registrar.grid(row=0, column=2, padx=5)

        btn_reg_webcam = tk.Button(frame_botones, text="Registrar con Webcam", width=18,
                                   command=self.registrar_rostro_webcam)
        btn_reg_webcam.grid(row=0, column=3, padx=5)

        btn_webcam = tk.Button(frame_botones, text="Webcam en tiempo real", width=18, command=self.abrir_webcam)
        btn_webcam.grid(row=0, column=4, padx=5)

    # Cargar imagen desde archivo
    def cargar_imagen(self):
        global RUTA_ULTIMA_IMAGEN
        file_path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg;*.jpeg;*.png")]
        )
        if not file_path:
            return
        RUTA_ULTIMA_IMAGEN = file_path

        img = Image.open(file_path)
        img = img.resize((400, 400))
        self.tk_img = ImageTk.PhotoImage(img)
        self.label_imagen.config(image=self.tk_img, text="")
        self.label_pred.config(text="Predicción: -")

    # Predicción con modelo
    def predecir(self):
        global RUTA_ULTIMA_IMAGEN
        if not RUTA_ULTIMA_IMAGEN:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        nombre, img_box = predecir_rostro(RUTA_ULTIMA_IMAGEN)
        self.label_pred.config(text=f"Predicción: {nombre}")

        if img_box is not None:
            img_rgb = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb).resize((400, 400))
            self.tk_img = ImageTk.PhotoImage(pil_img)
            self.label_imagen.config(image=self.tk_img, text="")

    # Registrar imagen manual
    def registrar_rostro(self):
        global RUTA_ULTIMA_IMAGEN
        if not RUTA_ULTIMA_IMAGEN:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        nombre = simpledialog.askstring("Nuevo rostro", "Nombre de la persona:")
        if not nombre:
            return

        try:
            agregar_rostro(RUTA_ULTIMA_IMAGEN, nombre)
            ok = reentrenar_clasificador()
            if ok:
                messagebox.showinfo("Éxito", f"Rostro '{nombre}' registrado y modelo actualizado.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo registrar el rostro:\n{e}")

    # NUEVO → Registrar rostro con webcam
    def registrar_rostro_webcam(self):
        script = "registrar_webcam.py"
        if not os.path.exists(script):
            messagebox.showerror("Error", f"No se encontró {script}")
            return

        subprocess.Popen(["python", script])

    # Webcam en tiempo real
    def abrir_webcam(self):
        script = "webcam_realtime.py"
        if not os.path.exists(script):
            messagebox.showerror("Error", f"No se encontró {script}")
            return

        subprocess.Popen(["python", script])


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
