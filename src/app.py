from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
from flask_cors import CORS 
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(os.path.join(BASE_DIR, 'models'))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
CORS(app)

try:
    print("Cargando modelo desde archivo...")
    modelo_path = os.path.join(BASE_DIR, 'modelo_rf.pkl')
    encoders_path = os.path.join(BASE_DIR, 'label_encoders.pkl')
    
    with open(modelo_path, 'rb') as f:
        modelo = pickle.load(f)
    
    with open(encoders_path, 'rb') as f:
        label_encoders = pickle.load(f)
    
    print("Modelo cargado correctamente")
except FileNotFoundError as e:
    print(f"Error: Archivos del modelo no encontrados. {e}")
    modelo = None
    label_encoders = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if modelo is None or label_encoders is None:
        return jsonify({'error': 'El modelo no está disponible'}), 500
    
    try:
        data = request.get_json()
        
        required_fields = ['brand', 'model', 'vehicle_type', 'gearbox', 'fuel_type', 'power', 'mileage', 'car_age', 'not_repaired']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Campo faltante: {field}'}), 400
        
        input_data = {}
        categorical_features = ['brand', 'model', 'vehicle_type', 'gearbox', 'fuel_type']
        
        for feature in categorical_features:
            if feature in data:
                le = label_encoders[feature]
                try:
                    if data[feature] in le.classes_:
                        input_data[feature] = le.transform([data[feature]])[0]
                    else:
                        input_data[feature] = 0
                except Exception as e:
                    print(f"Error codificando {feature}: {e}")
                    input_data[feature] = 0
        
        try:
            input_data['power'] = float(data['power'])
            input_data['mileage'] = float(data['mileage'])
            input_data['car_age'] = float(data['car_age'])
            input_data['not_repaired'] = int(data['not_repaired'])
        except ValueError as e:
            return jsonify({'error': f'Error en datos numéricos: {e}'}), 400
        
        feature_order = ['brand', 'model', 'power', 'mileage', 'vehicle_type', 'gearbox', 'fuel_type', 'car_age', 'not_repaired']
        X_array = np.array([[input_data[feature] for feature in feature_order]])
        
        import pandas as pd
        X_df = pd.DataFrame(X_array, columns=feature_order)
        
        predicted_price = modelo.predecir(X_df)[0]
        usd_price = predicted_price * 1.1
        
        return jsonify({
            'eur': round(predicted_price, 2),
            'usd': round(usd_price, 2)
        })
        
    except Exception as e:
        print(f"Error en la predicción: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
