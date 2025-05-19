import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Deshabilitar GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reducir mensajes de log

# Importa TensorFlow después de configurar las variables de entorno
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Asegurar que no se use GPU
# Limitar el uso de memoria
physical_devices = tf.config.list_physical_devices('CPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

from flask import Flask, request, jsonify, render_template
import numpy as np
import traceback

app = Flask(__name__)

# Variables globales para estadísticas
AGE_MEAN = 29.7
AGE_STD = 14.5
FARE_MEAN = 32.2
FARE_STD = 49.7

# Variable global para el estado del modelo
model = None
model_status = "Model not loaded yet"

# Configuración global para manejo de errores
@app.errorhandler(Exception)
def handle_error(error):
    response = {
        "error": str(error),
        "status": "error"
    }
    return jsonify(response), 500

def load_model_lazy():
    """Carga el modelo solo cuando sea necesario"""
    global model, model_status
    if model is None:
        try:
            from tensorflow.keras.models import load_model
            model = load_model('model/titanic_model.h5', compile=False)
            model_status = "Model loaded successfully"
        except Exception as e:
            model_status = f"Error loading model: {str(e)}"
            model = None
    return model

def preprocess_input(age, fare, gender, embarked):
    age_norm = (age - AGE_MEAN) / AGE_STD
    fare_norm = (fare - FARE_MEAN) / FARE_STD
    
    gender_factor = 0.2 if gender == 1 else 0.8  
    
    fare_factor = 0.3 if fare < 10 else 0.7
    
    embarked_factor = 0.4 if embarked == 2 else 0.6
    
    if gender == 1 and fare < 10:
        fare_factor *= 0.5  
    
    base_probability = (gender_factor + fare_factor + embarked_factor) / 3
    
    x = np.array([[age_norm, fare_norm, gender, embarked]], dtype=float)
    
    # Cargar el modelo solo cuando sea necesario
    current_model = load_model_lazy()
    if current_model is None:
        return base_probability  # Fallback si el modelo no está disponible
        
    # Usar modo de predicción optimizado
    with tf.device('/CPU:0'):
        model_prob = current_model.predict(x, verbose=0)[0][0]
    
    if gender == 1 and fare < 10:
        final_prob = (model_prob * 0.3 + base_probability * 0.7)
    else:
        final_prob = (model_prob * 0.5 + base_probability * 0.5)
    
    return final_prob

@app.route('/')
def index():
    return render_template('index.html', model_status=model_status)

@app.route('/model-status')
def get_model_status():
    return jsonify({'status': model_status})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or request.form
        if not data:
            return jsonify({
                'error': 'No data provided',
                'status': 'error'
            }), 400

        age = float(data.get('age', 0))
        fare = float(data.get('fare', 0))
        gender = int(data.get('gender', 0))
        embarked = int(data.get('embarked', 0))
        
        if not (0 <= age <= 100):
            return jsonify({
                'error': 'Age must be between 0 and 100',
                'status': 'error'
            }), 400
        if fare < 0:
            return jsonify({
                'error': 'Fare must be positive',
                'status': 'error'
            }), 400
        if gender not in [0, 1]:
            return jsonify({
                'error': 'Gender must be 0 or 1',
                'status': 'error'
            }), 400
        if embarked not in [0, 1, 2]:
            return jsonify({
                'error': 'Embarked must be 0, 1, or 2',
                'status': 'error'
            }), 400

        probability = preprocess_input(age, fare, gender, embarked)
        label = int(probability > 0.5)
        
        return jsonify({
            'probability': float(probability),
            'survived': label,
            'status': 'success',
            'raw_input': {
                'age': age,
                'fare': fare,
                'gender': gender,
                'embarked': embarked
            }
        })
    except Exception as e:
        app.logger.error(f"Error in predict: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
