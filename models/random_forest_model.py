import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

class ArbolDecision:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def dividir_nodo(self, X, y, profundidad):
        m = X.shape[1]
        if len(y) <= 3 or profundidad >= self.max_depth:
            return np.mean(y)

        n_features = int(np.sqrt(m))
        features = np.random.choice(m, n_features, replace=False)

        mejor_ganancia = 0
        mejor_pregunta = None
        mejor_izquierda = None
        mejor_derecha = None

        for col in features:
            valores_unicos = np.unique(X[:, col])
            for valor in valores_unicos:
                pregunta = (col, valor)
                izquierda, derecha = self.partir(X, y, pregunta)
                if len(izquierda[1]) > 0 and len(derecha[1]) > 0:
                    ganancia = self.ganancia_informacion(y, izquierda[1], derecha[1])
                    if ganancia > mejor_ganancia:
                        mejor_ganancia = ganancia
                        mejor_pregunta = pregunta
                        mejor_izquierda = izquierda
                        mejor_derecha = derecha

        if mejor_ganancia > 0:
            izquierda = self.dividir_nodo(mejor_izquierda[0], mejor_izquierda[1], profundidad + 1)
            derecha = self.dividir_nodo(mejor_derecha[0], mejor_derecha[1], profundidad + 1)
            return (mejor_pregunta, izquierda, derecha)

        return np.mean(y)

    def partir(self, X, y, pregunta):
        col, valor = pregunta
        mascara = X[:, col] >= valor
        return (X[~mascara], y[~mascara]), (X[mascara], y[mascara])

    def ganancia_informacion(self, padre, izquierda, derecha):
        p = len(izquierda) / len(padre)
        return self.varianza(padre) - p * self.varianza(izquierda) - (1 - p) * self.varianza(derecha)

    def varianza(self, y):
        if len(y) == 0:
            return 0
        return np.var(y)

    def ajustar(self, X, y):
        self.raiz = self.dividir_nodo(X.values, y.values, 0)

    def predecir_uno(self, x, nodo):
        if isinstance(nodo, tuple):
            pregunta, izquierda, derecha = nodo
            if x[pregunta[0]] >= pregunta[1]:
                return self.predecir_uno(x, derecha)
            else:
                return self.predecir_uno(x, izquierda)
        else:
            return nodo

    def predecir(self, X):
        return [self.predecir_uno(x, self.raiz) for x in X.values]

class BosqueAleatorio:
    def __init__(self, n_arboles=10, max_depth=5):
        self.n_arboles = n_arboles
        self.max_depth = max_depth
        self.arboles = []

    def ajustar(self, X, y):
        for _ in range(self.n_arboles):
            arbol = ArbolDecision(max_depth=self.max_depth)
            indices = np.random.choice(len(X), len(X), replace=True)
            X_muestra = X.iloc[indices]
            y_muestra = y.iloc[indices]
            arbol.ajustar(X_muestra, y_muestra)
            self.arboles.append(arbol)

    def predecir(self, X):
        predicciones = np.array([arbol.predecir(X) for arbol in self.arboles])
        return np.mean(predicciones, axis=0)

def cargar_y_procesar_datos():
    print("Cargando datos...")
    
    ruta = './data/car_data.csv'
    print(f"Cargando desde: {ruta}")
    df = pd.read_csv(ruta)
    print(f"¡Datos cargados exitosamente desde {ruta}!")
    
    print("Procesando datos...")
    df = df.drop_duplicates().reset_index(drop=True)
    df.columns = df.columns.str.replace(r'([A-Z])', r'_\1', regex=True).str.strip('_').str.lower()
    df_filtered = df.query('1900 <= registration_year <= 2016')
    df_filtered = df_filtered.query('99 <= price')
    df_filtered = df_filtered.query('power <= 2000')
    df_filtered.loc[df_filtered['power'] < 50, 'power'] = np.nan
    df_final = df_filtered.drop(['date_crawled', 'registration_year', 'registration_month', 'date_created', 'postal_code', 'last_seen'], axis=1)
    
    return df_final

def entrenar_modelo():
    df_model = cargar_y_procesar_datos()
    
    label_encoders = {}
    for col in df_model.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    X = df_model.drop('price', axis=1)
    y = df_model['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Entrenando modelo Random Forest...")
    modelo_rf = BosqueAleatorio(n_arboles=10, max_depth=5)
    modelo_rf.ajustar(X_train, y_train)

    y_pred = modelo_rf.predecir(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    print(f"Rendimiento del modelo:")
    print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")
    print(f"R-cuadrado (R²): {r2:.4f}")

    print("Modelo entrenado y listo para ser guardado.")
    
    return modelo_rf, label_encoders

if __name__ == "__main__":
    modelo_rf, label_encoders = entrenar_modelo()
