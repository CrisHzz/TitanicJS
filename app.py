import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Deshabilitar GPU
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Asegurar que no se use GPU
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
try:
    model = load_model('model/titanic_model.h5', compile=False)  # No compilar el modelo
    model_status = "Model loaded successfully"
except Exception as e:
    model_status = f"Error loading model: {str(e)}"
    model = None

AGE_MEAN = 29.7
AGE_STD = 14.5
FARE_MEAN = 32.2
FARE_STD = 49.7

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
    
    model_prob = model.predict(x)[0][0]
    
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
    data = request.json or request.form
    try:
        age = float(data['age'])
        fare = float(data['fare'])
        gender = int(data['gender'])
        embarked = int(data['embarked'])
        
        if not (0 <= age <= 100):
            return jsonify({'error': 'Age must be between 0 and 100'}), 400
        if fare < 0:
            return jsonify({'error': 'Fare must be positive'}), 400
        if gender not in [0, 1]:
            return jsonify({'error': 'Gender must be 0 or 1'}), 400
        if embarked not in [0, 1, 2]:
            return jsonify({'error': 'Embarked must be 0, 1, or 2'}), 400

        probability = preprocess_input(age, fare, gender, embarked)
        label = int(probability > 0.5)
        
        return jsonify({
            'probability': float(probability),
            'survived': label,
            'raw_input': {
                'age': age,
                'fare': fare,
                'gender': gender,
                'embarked': embarked
            }
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
