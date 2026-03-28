import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── Load Model & Scaler ────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE_DIR, 'model', 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')

model  = None
scaler = None

def load_artifacts():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("[OK] Model & Scaler loaded successfully.")
    else:
        print("[WARN] Model files not found. Please run train_model.py first.")

load_artifacts()

# ─── Feature definitions for validation ─────────────────────────
FEATURES = [
    {"name": "Pregnancies",              "min": 0,    "max": 20},
    {"name": "Glucose",                  "min": 0,    "max": 300},
    {"name": "BloodPressure",            "min": 0,    "max": 180},
    {"name": "SkinThickness",            "min": 0,    "max": 100},
    {"name": "Insulin",                  "min": 0,    "max": 900},
    {"name": "BMI",                      "min": 0.0,  "max": 70.0},
    {"name": "DiabetesPedigreeFunction", "min": 0.0,  "max": 3.0},
    {"name": "Age",                      "min": 1,    "max": 120},
]

# ─── Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "version":      "1.0.0"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(force=True)

    # ── Extract & validate ──
    try:
        features = [float(data.get(feat["name"], 0)) for feat in FEATURES]
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    # ── Scale & predict ──
    X = np.array([features])
    X_scaled = scaler.transform(X)

    prediction   = int(model.predict(X_scaled)[0])
    probability  = float(model.predict_proba(X_scaled)[0][1])  # prob of diabetes
    probability_pct = round(probability * 100, 1)

    # ── Risk level ──
    if probability < 0.35:
        risk_level = "Low Risk"
        risk_color = "#22c55e"
    elif probability < 0.65:
        risk_level = "Moderate Risk"
        risk_color = "#f59e0b"
    else:
        risk_level = "High Risk"
        risk_color = "#ef4444"

    return jsonify({
        "prediction":      prediction,
        "probability":     probability_pct,
        "risk_level":      risk_level,
        "risk_color":      risk_color,
        "has_diabetes":    prediction == 1,
        "confidence":      round(max(probability, 1 - probability) * 100, 1)
    })

@app.route('/features')
def get_features():
    return jsonify(FEATURES)

# ─── Entry Point ────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
