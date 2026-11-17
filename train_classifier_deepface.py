# train_classifier_deepface.py
import json
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

EMB_FILE = "modelo/embeddings.json"

def train_classifier():
    with open(EMB_FILE, "r") as f:
        data = json.load(f)

    X = [d["embedding"] for d in data]
    y_names = [d["name"] for d in data]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_names)

    clf = SVC(kernel="linear", probability=True)
    clf.fit(X, y)

    joblib.dump(clf, "modelo/clasificador.pkl")
    joblib.dump(encoder, "modelo/encoder.pkl")

    print("🎯 Clasificador entrenado correctamente.")
    print(f"Total personas: {len(set(y_names))}")
    print(f"Total embeddings: {len(X)}")


if __name__ == "__main__":
    train_classifier()
