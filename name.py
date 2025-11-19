from sklearn.datasets import fetch_lfw_people
lfw = fetch_lfw_people(min_faces_per_person=60)
print(len(lfw.target_names))
print(lfw.target_names)
