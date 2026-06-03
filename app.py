from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model artifacts once when server starts
# We load them here (not inside the route) so we
# don't reload from disk on every single request

model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')
threshold = joblib.load('best_threshold.pkl')
print("Model loaded")
print("Scaler loaded")
print(f"Threshold loaded: {threshold:.4f}")

# Route 1: Home page — serves the UI
@app.route('/')
def home():
    return render_template('index.html')


# Route 2: Health check — useful for deployment
# Just hit /health to confirm the API is running
@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "threshold": round(float(threshold), 4)
    })


# Route 3: Predict — the core endpoint
#
# Expects a POST request with JSON body:
# {
#   "Time": 0.0,
#   "V1": -1.35, "V2": -0.07, ..., "V28": -0.02,
#   "Amount": 149.62
# }
#
# Returns:
# {
#   "fraud": true/false,
#   "probability": 0.94,
#   "threshold_used": 0.3721,
#   "verdict": "FRAUD DETECTED" / "LEGITIMATE"
# }
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        expected_features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        missing_features = [f for f in expected_features if f not in data]
        if missing_features:
            return jsonify({
                "error": f"Missing features: {', '.join(missing_features)}"
            }), 400

        # --- Build feature array in correct order --- must match what the model was trained on
        feature_values = [data[f] for f in expected_features]
        input_array = np.array(feature_values).reshape(1, -1)

        # --- Scale features ---
        input_array[:, [0, 29]] = scaler.transform(input_array[:, [0, 29]]) # These were scaled during training so we must scale them here too Columns 0 = Time, column 29 = Amount (same positions as training)

        # --- Predict probability ---
        probability = model.predict_proba(input_array)[0][1]  # probability of fraud
        is_fraud    = bool(probability >= threshold)

        return jsonify({
            "fraud": is_fraud,
            "probability": round(float(probability), 4),
            "threshold_used": round(float(threshold), 4),
            "verdict": "FRAUD DETECTED" if is_fraud else "LEGITIMATE"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Route 4: Sample transaction — useful for testing the UI and API without needing to input real data (we use a real fraudulent transaction from the creditcard dataset) 
@app.route('/sample')
def sample():
    # This is a real fraudulent transaction from the creditcard dataset
    sample_transaction = {
        "Time": 406.0,
        "V1": -2.3122265423263, "V2": 1.95199201064158,
        "V3": -1.60985073229769, "V4": 3.9979055875468,
        "V5": -0.522187864667764, "V6": -1.42654531920595,
        "V7": -2.53738730624579, "V8": 1.39165724829804,
        "V9": -2.77008927719433, "V10": -2.77227214465915,
        "V11": 3.20203320709635, "V12": -2.89990738849473,
        "V13": -0.595221881324605, "V14": -4.28925378244217,
        "V15": 0.389724865343203, "V16": -1.14074717980657,
        "V17": -2.83005567450437, "V18": -0.0168224681808257,
        "V19": 0.416955705037907, "V20": 0.126910559061474,
        "V21": 0.517232370861764, "V22": -0.0350493686207606,
        "V23": -0.465211076182388, "V24": 0.320198198514526,
        "V25": 0.0445191674731724, "V26": 0.177839798284401,
        "V27": 0.261145002567677, "V28": -0.143275874698919,
        "Amount": 239.93
    }
    return jsonify(sample_transaction)


if __name__ == '__main__':
    app.run(debug=True, port=5000)