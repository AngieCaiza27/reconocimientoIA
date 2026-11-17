import json

ruta = "modelo/embeddings.json"

with open(ruta, "r", encoding="utf-8") as f:
    data = json.load(f)

# Filtrar todo lo que NO sea "Angie Caiza"
nuevo = [item for item in data if item.get("name") != "Angie Caiza"]

with open(ruta, "w", encoding="utf-8") as f:
    json.dump(nuevo, f, indent=2)

print("✅ Se eliminaron todos los registros 'Angie Caiza' sin tocar el resto del dataset.")
