from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Linear regression coefficients for Li-ion battery efficiency
COEFFICIENTS = {
    'intercept': 97.5,
    'soc_abs': 0.0015,
    'voltage': -0.05,
    'current': -0.002,
    'battery_temp': -0.12,
    'ambient_temp': -0.08
}

@app.route('/api/predict', methods=['POST'])
def predict_efficiency():
    """Predict efficiency based on battery metrics"""
    try:
        data = request.json
        
        soc_abs = abs(float(data.get('soc', 0)))
        voltage = float(data.get('voltage', 0))
        current = abs(float(data.get('current', 0)))
        battery_temp = 43
        ambient_temp = float(data.get('ambient_temp', 25))
        
        prediction = (
            COEFFICIENTS['intercept'] +
            COEFFICIENTS['soc_abs'] * soc_abs +
            COEFFICIENTS['voltage'] * voltage +
            COEFFICIENTS['current'] * current +
            COEFFICIENTS['battery_temp'] * battery_temp +
            COEFFICIENTS['ambient_temp'] * ambient_temp
        )
        
        prediction = max(min(prediction, 100), 0)
        
        return jsonify({
            'efficiency': round(prediction, 5),
            'input': {
                'soc_abs': round(soc_abs, 2),
                'voltage': round(voltage, 3),
                'current': round(current, 3),
                'battery_temp': round(battery_temp, 1),
                'ambient_temp': round(ambient_temp, 1)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'Linear Regression (Li-ion)'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("EV WPT BATTERY EFFICIENCY API ⚡")
    print("="*60)
    print("\nRunning on: http://localhost:5000")
    print("Endpoint: POST /api/predict")
    print("Press CTRL+C to stop\n")
    app.run(debug=True, port=5000)

@app.route('/api/predict', methods=['POST'])
def predict_efficiency():
    """
    Predict efficiency based on battery metrics
    Expected JSON:
    {
        "soc": 45.5,
        "voltage": 3.8,
        "current": 25.0,
        "battery_temp": 30.5,
        "ambient_temp": 25.0
    }
    """
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Extract features
        soc_abs = abs(float(data.get('soc', 0)))
        voltage = float(data.get('voltage', 0))
        current = float(data.get('current', 0))
        battery_temp = 43
        ambient_temp = float(data.get('ambient_temp', 0))
        
        # Create feature array in correct order
        features = np.array([[soc_abs, voltage, current, battery_temp, ambient_temp]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # Ensure prediction is within valid range
        prediction = max(min(prediction, 100), 0)
        
        return jsonify({
            'efficiency': round(prediction, 5),
            'input': {
                'soc_abs': soc_abs,
                'voltage': voltage,
                'current': current,
                'battery_temp': battery_temp,
                'ambient_temp': ambient_temp
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
