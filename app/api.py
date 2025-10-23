from flask import Flask, request, jsonify
import numpy as np
from src.model_utils import load_model

app = Flask(__name__)
model = load_model()

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        pred = model.predict(features)[0]
        return jsonify({"prediction": float(round(pred, 4))})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "✅ API is running!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)