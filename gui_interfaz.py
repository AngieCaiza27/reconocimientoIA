# gui_interfaz.py (versión corregida con diagnóstico)
import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2

from predict_hibrido import predecir_rostro_hibrido
from model_utils import agregar_rostro_hibrido, reentrenar_hibrido

RUTA_ULTIMA_IMAGEN = None


class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento Facial - Modelo Híbrido")
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

    def cargar_imagen(self):
        global RUTA_ULTIMA_IMAGEN
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Todos", "*.*")
            ]
        )
        if not file_path:
            return

        # Normalizar ruta
        RUTA_ULTIMA_IMAGEN = os.path.normpath(file_path)
        
        print(f"\n{'='*60}")
        print(f"IMAGEN CARGADA")
        print(f"Ruta: {RUTA_ULTIMA_IMAGEN}")
        print(f"Existe: {os.path.exists(RUTA_ULTIMA_IMAGEN)}")
        
        # Verificar con OpenCV
        test_img = cv2.imread(RUTA_ULTIMA_IMAGEN)
        if test_img is None:
            print("❌ No se puede leer con cv2.imread()")
            messagebox.showerror("Error", "No se pudo leer la imagen")
            return
        print(f"✓ Imagen válida: {test_img.shape}")
        print(f"{'='*60}\n")

        img = Image.open(RUTA_ULTIMA_IMAGEN).resize((400, 400))
        self.tk_img = ImageTk.PhotoImage(img)
        self.label_imagen.config(image=self.tk_img, text="")
        self.label_pred.config(text="Predicción: -")

    def predecir(self):
        global RUTA_ULTIMA_IMAGEN
        if not RUTA_ULTIMA_IMAGEN:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        print(f"\n{'='*60}")
        print(f"PREDICIENDO")
        print(f"Ruta: {RUTA_ULTIMA_IMAGEN}")
        
        nombre, rostro = predecir_rostro_hibrido(RUTA_ULTIMA_IMAGEN)
        
        print(f"Resultado: {nombre}")
        print(f"Rostro: {rostro is not None}")
        print(f"{'='*60}\n")
        
        self.label_pred.config(text=f"Predicción: {nombre}")

        if rostro is not None:
            rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rostro_rgb).resize((400, 400))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.label_imagen.config(image=self.tk_img, text="")
        else:
            messagebox.showinfo("Info", f"Resultado: {nombre}")

    def registrar_rostro(self):
        global RUTA_ULTIMA_IMAGEN
        if not RUTA_ULTIMA_IMAGEN:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return

        nombre = simpledialog.askstring("Nuevo rostro", "Nombre de la persona:")
        if not nombre:
            return

        try:
            agregar_rostro_hibrido(RUTA_ULTIMA_IMAGEN, nombre)
            ok = reentrenar_hibrido()

            if ok:
                messagebox.showinfo("Éxito", f"Rostro '{nombre}' registrado.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al registrar:\n{e}")

    def registrar_rostro_webcam(self):
        if not os.path.exists("registrar_webcam.py"):
            messagebox.showerror("Error", "No se encontró registrar_webcam.py")
            return
        subprocess.Popen(["python", "registrar_webcam.py"])

    def abrir_webcam(self):
        if not os.path.exists("webcam_realtime.py"):
            messagebox.showerror("Error", "No se encontró webcam_realtime.py")
            return
        subprocess.Popen(["python", "webcam_realtime.py"])


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()