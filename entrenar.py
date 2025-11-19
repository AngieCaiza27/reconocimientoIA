# entrenar.py (parametrizable)
import os
import json
import argparse
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar clasificador LFW (Eigenfaces + SVM)")
    parser.add_argument("--min-faces", type=int, default=50, help="Mínimo de fotos por persona en LFW (reduce para incluir más identidades)")
    parser.add_argument("--resize", type=float, default=0.5, help="Factor de resize al cargar LFW (0.5 → 62x47)")
    parser.add_argument("--top-n", type=int, default=None, help="Opcional: mantener solo las N identidades con más imágenes")
    parser.add_argument("--pca-components", type=int, default=150, help="Número de componentes PCA")
    parser.add_argument("--test-size", type=float, default=0.25, help="Proporción del conjunto de test")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla aleatoria")
    return parser.parse_args()


def main():
    args = parse_args()

    # Crear carpeta modelo si no existe
    os.makedirs("modelo", exist_ok=True)

    print("Cargando dataset LFW...")
    lfw = fetch_lfw_people(min_faces_per_person=args.min_faces, resize=args.resize)
    X = lfw.data
    y = lfw.target
    names = lfw.target_names  # numpy array de strings

    print("Imágenes:", X.shape)
    print("Clases detectadas:", len(names))

    # Si se solicita top-N, filtramos por las identidades con más ejemplos
    if args.top_n is not None and args.top_n > 0:
        counts = np.bincount(y, minlength=len(names))
        # índices ordenados por frecuencia desc (más a menos)
        top_indices = np.argsort(counts)[::-1][: args.top_n]
        top_set = set(top_indices.tolist())
        mask = np.array([label in top_set for label in y])

        X = X[mask]
        y_old = y[mask]
        # Remapeo de etiquetas a [0..top_n-1]
        old_to_new = {old: new for new, old in enumerate(top_indices)}
        y = np.array([old_to_new[int(lbl)] for lbl in y_old])
        names = names[top_indices]

        print(f"Aplicado top-N={args.top_n}. Nuevas clases: {len(names)}. Nuevas imágenes: {X.shape[0]}")

    # PREPROCESAMIENTO
    print("Normalizando datos...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Aplicando PCA (Eigenfaces)...")
    pca = PCA(n_components=args.pca_components, whiten=True, random_state=args.random_state)
    X_pca = pca.fit_transform(X_scaled)

    # CLASIFICADOR
    print("Entrenando clasificador SVM...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=args.test_size, random_state=args.random_state
    )

    clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=args.random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=names))

    # GUARDAR MODELOS
    print("\nGuardando modelos...")
    joblib.dump(clf, "modelo/clasificador.pkl")
    joblib.dump(pca, "modelo/pca.pkl")
    joblib.dump(scaler, "modelo/scaler.pkl")
    joblib.dump(names, "modelo/nombres.pkl")

    # Guardar metadatos del entrenamiento para trazabilidad
    metadata = {
        "min_faces_per_person": args.min_faces,
        "resize": args.resize,
        "top_n": args.top_n,
        "pca_components": args.pca_components,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "n_classes": int(len(names)),
        "n_samples": int(X.shape[0]),
    }
    with open("modelo/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nEntrenamiento completado correctamente.")


if __name__ == "__main__":
    main()
