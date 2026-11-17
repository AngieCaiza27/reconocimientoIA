# train_classifier_cv.py
import json
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

EMB_FILE = "modelo/embeddings.json"

def load_data():
    with open(EMB_FILE, "r") as f:
        data = json.load(f)

    X = np.array([d["embedding"] for d in data])
    y_names = [d["name"] for d in data]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_names)

    return X, y, encoder, y_names

def main():
    X, y, encoder, y_names = load_data()

    print(f"Total muestras: {len(X)} | Personas: {len(set(y_names))}")

    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }

    svm = SVC(probability=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        svm,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X, y)

    print("Mejores hiperparámetros:", grid.best_params_)
    print("Mejor accuracy promedio CV:", grid.best_score_)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X)
    print("\nAccuracy sobre todo el conjunto:", accuracy_score(y, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y, y_pred, target_names=encoder.classes_))

    joblib.dump(best_model, "modelo/clasificador.pkl")
    joblib.dump(encoder, "modelo/encoder.pkl")
    print("\n✅ Modelo con mejores hiperparámetros guardado.")


if __name__ == "__main__":
    main()
