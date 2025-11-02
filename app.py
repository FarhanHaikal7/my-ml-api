from flask import Flask, request, jsonify
import joblib
import numpy as np


app = Flask( __name__)

# Load the trained model and scaler
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')

except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return "Ml model API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Model or scaler not loaded properly."}), 500
    
    try:
        data = request.get_json()

        features = data['features']

        if len(features) != 30:
            return jsonify({'error': f'Expected 30 features for prediction, got {len(features)}'}), 400
        
        features_array = np.array(features).reshape(1, -1)

        scaled_features = scaler.transform(features_array)

        prediction = model.predict(scaled_features)

        probalities = model.predict_proba(scaled_features)

        prediction_label = 'benign' if prediction[0] == 1 else 'malignant'
        confidence = probalities[0][prediction[0]]

        return jsonify({
            'prediction': int(prediction[0]),
            'label': prediction_label,
            'confidence': float(confidence)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
if __name__ == '__main__':
        app.run(debug=True, host='0.0.0.0', port=5000)




        