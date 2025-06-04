import pickle
from random_forest_model import entrenar_modelo

print("Iniciando entrenamiento del modelo...")
modelo_rf, label_encoders = entrenar_modelo()

print("Guardando modelo en disco...")
with open('./modelo_rf.pkl', 'wb') as f:
    pickle.dump(modelo_rf, f)

with open('./label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Modelo guardado correctamente")
