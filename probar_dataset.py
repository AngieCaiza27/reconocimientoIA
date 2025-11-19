from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

# Cargar dataset LFW
lfw = fetch_lfw_people(min_faces_per_person=50, resize=0.5)

# Mostrar la primera imagen y su nombre
plt.imshow(lfw.images[0], cmap="gray")
plt.title(lfw.target_names[lfw.target[0]])
plt.axis("off")
plt.show()
