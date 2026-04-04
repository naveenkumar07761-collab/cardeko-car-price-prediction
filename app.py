"""
Cardeko Car Price Prediction — Flask REST API
Trained on real Cardekho dataset (15,331 cars, 32 brands)

Endpoints:
  GET  /             → Web UI
  POST /predict      → JSON price prediction
  GET  /models/<brand> → Models for a brand (for dynamic dropdowns)
  GET  /model-info   → Model metadata & metrics
  GET  /health       → Health check
"""

from flask import Flask, request, jsonify, render_template
import pickle, numpy as np, os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'cardeko_rf_model.pkl')
artifacts  = None

# ── Load Model ────────────────────────────────────────────
def load_model():
    global artifacts
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run `python train_model.py` first.")
    with open(MODEL_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    print(f"✅ Model loaded  R²={artifacts['metrics']['r2']}  "
          f"MAE=₹{artifacts['metrics']['mae']:,.0f}")

# ── Feature Engineering ───────────────────────────────────
def engineer_features(d: dict) -> dict:
    km      = float(d['km_driven'])
    age     = float(d['vehicle_age'])
    engine  = float(d['engine'])
    power   = float(d['max_power'])
    mileage = float(d['mileage'])

    d['km_per_year']       = km / (age + 1)
    d['power_to_engine']   = power / engine
    d['mileage_per_power'] = mileage / (power + 1)
    d['price_segment']     = (
        0 if engine <= 1000 else
        1 if engine <= 1500 else
        2 if engine <= 2000 else
        3 if engine <= 3000 else 4
    )
    return d

# ── Encode Categoricals ───────────────────────────────────
def encode_input(d: dict) -> dict:
    for col in artifacts['cat_cols']:
        le  = artifacts['label_encoders'][col]
        val = str(d[col])
        if val not in le.classes_:
            val = le.classes_[0]          # fallback to first known class
        d[col + '_enc'] = int(le.transform([val])[0])
    return d

# ── Build Feature Vector ──────────────────────────────────
def build_vector(d: dict) -> np.ndarray:
    return np.array([[float(d[c]) for c in artifacts['feature_cols']]])

# ── Validation ────────────────────────────────────────────
REQUIRED = ['brand', 'model', 'vehicle_age', 'km_driven',
            'seller_type', 'fuel_type', 'transmission_type',
            'mileage', 'engine', 'max_power', 'seats']

def validate(d: dict):
    missing = [f for f in REQUIRED if f not in d]
    if missing:
        return False, f"Missing fields: {missing}"
    if not (0 <= int(d['vehicle_age']) <= 35):
        return False, "vehicle_age must be 0–35"
    if float(d['km_driven']) < 0:
        return False, "km_driven must be >= 0"
    if float(d['max_power']) <= 0:
        return False, "max_power must be > 0"
    if float(d['engine']) <= 0:
        return False, "engine must be > 0"
    if not (1 <= int(d['seats']) <= 10):
        return False, "seats must be 1–10"
    return True, None

# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────

@app.route('/')
def index():
    a = artifacts or {}
    return render_template('index.html',
        brands        = a.get('brands', []),
        brand_model_map = a.get('brand_model_map', {}),
        fuel_types    = a.get('fuel_types', []),
        seller_types  = a.get('seller_types', []),
        transmissions = a.get('transmissions', []),
        metrics       = a.get('metrics', {}),
    )


@app.route('/models/<brand>')
def get_models(brand):
    """Dynamic dropdown: GET /models/Hyundai → list of models"""
    brand_map = (artifacts or {}).get('brand_model_map', {})
    # Try exact match, then title-cased
    models = brand_map.get(brand) or brand_map.get(brand.title()) or []
    return jsonify({'brand': brand, 'models': models})


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    {
      "brand": "Hyundai", "model": "i20",
      "vehicle_age": 5, "km_driven": 40000,
      "seller_type": "Dealer", "fuel_type": "Petrol",
      "transmission_type": "Manual",
      "mileage": 18.9, "engine": 1197,
      "max_power": 82.0, "seats": 5
    }
    """
    if artifacts is None:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': 'Invalid JSON'}), 400

    ok, err = validate(data)
    if not ok:
        return jsonify({'error': err}), 400

    try:
        d = dict(data)
        d = engineer_features(d)
        d = encode_input(d)
        X = build_vector(d)

        rf            = artifacts['model']
        predicted     = float(rf.predict(X)[0])
        predicted     = max(40000, round(predicted, -3))   # round to nearest ₹1000

        # Confidence interval via tree std-dev
        tree_preds    = np.array([t.predict(X)[0] for t in rf.estimators_])
        std           = float(np.std(tree_preds))
        low           = round(max(40000, predicted - 1.5 * std), -3)
        high          = round(predicted + 1.5 * std, -3)

        cv_val = std / predicted
        if cv_val < 0.08:   confidence = "Very High"
        elif cv_val < 0.16: confidence = "High"
        elif cv_val < 0.28: confidence = "Medium"
        else:               confidence = "Low"

        return jsonify({
            'predicted_price':    predicted,
            'predicted_lakhs':    round(predicted / 100000, 2),
            'price_range': {
                'low':       low,
                'high':      high,
                'low_lakhs': round(low / 100000, 2),
                'high_lakhs':round(high / 100000, 2),
            },
            'confidence':    confidence,
            'input_summary': {
                'brand':        d['brand'],
                'model':        d['model'],
                'vehicle_age':  int(d['vehicle_age']),
                'fuel_type':    d['fuel_type'],
                'transmission': d['transmission_type'],
                'km_driven':    int(d['km_driven']),
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model-info')
def model_info():
    if not artifacts:
        return jsonify({'error': 'Model not loaded'}), 503
    rf = artifacts['model']
    return jsonify({
        'algorithm':    'Random Forest Regressor',
        'n_estimators': rf.n_estimators,
        'max_depth':    rf.max_depth,
        'features':     artifacts['feature_cols'],
        'metrics':      artifacts['metrics'],
        'brands':       artifacts['brands'],
    })


@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': artifacts is not None})


if __name__ == '__main__':
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"⚠  {e}\n   Training now...")
        import subprocess, sys
        subprocess.run([sys.executable, 'train_model.py'], check=True)
        load_model()

    print("\n🚗 Cardeko API  →  http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
