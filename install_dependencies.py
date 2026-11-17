import os
import subprocess
import sys

def run(cmd):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

print("🔄 Eliminando venv si existe...")
run("rmdir /s /q venv")

print("🔧 Creando nuevo entorno virtual...")
run(f"{sys.executable} -m venv venv")

pip = r".\venv\Scripts\pip.exe"

packages = [
    "numpy==1.24.3",
    "tensorflow==2.13.0",
    "keras==2.13.1",
    "protobuf==4.23.4",
    "tensorboard==2.13.0",
    "typing-extensions==4.5.0",
    "opencv-python",
    "pillow",
    "retina-face",
    "deepface==0.0.93",
    "scikit-learn",
    "mtcnn"
]

print("\n📦 Instalando paquetes compatibles:")
for pkg in packages:
    run(f"{pip} install {pkg}")

print("\n🎉 ¡Entorno instalado correctamente!")
print("Activa con: .\\venv\\Scripts\\activate")
