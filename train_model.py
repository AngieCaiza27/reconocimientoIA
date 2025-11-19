# train_model.py
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

print("Cargando LFW...")
lfw = fetch_lfw_people(min_faces_per_person=60, resize=0.5)
X = lfw.data           #  n_samples x (62*47)
y = lfw.target         #  etiquetas
names = lfw.target_names

print("Dividiendo datos...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("Escalando datos...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Aplicando PCA...")
pca = PCA(n_components=150, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print("Entrenando SVM...")
clf = SVC(kernel='rbf', class_weight='balanced')
clf.fit(X_train_pca, y_train)

print("Guardando modelos...")
joblib.dump(clf,   "modelo/clasificador.pkl")
joblib.dump(pca,   "modelo/pca.pkl")
joblib.dump(scaler,"modelo/scaler.pkl")
joblib.dump(names, "modelo/nombres.pkl")

print("Accuracy:", clf.score(X_test_pca, y_test))
print("¡Modelo entrenado y guardado correctamente!")
