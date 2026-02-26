import json

with open("data/splits/ham_patient_split.json") as f:
    d = json.load(f)

print("Type:", type(d))

if isinstance(d, dict):
    print("Keys:", list(d.keys()))
    for k in d:
        print(k, "length:", len(d[k]))
else:
    print("Length:", len(d))