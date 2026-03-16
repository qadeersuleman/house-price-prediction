from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import pandas as pd

app = Flask(__name__)
project_root = Path(__file__).resolve().parent.parent
MODEL_PATH = project_root / 'src' / 'model.pkl'


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


model = load_model()


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.json
    df = pd.DataFrame([data])
    if model is None:
        return jsonify({'error': 'model not found, run src/train.py first'}), 500
    pred = model.predict(df)[0]
    return jsonify({'prediction': float(pred)})


if __name__ == '__main__':
    app.run(debug=True, port=8000)