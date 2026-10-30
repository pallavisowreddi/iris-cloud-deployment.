
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
data = joblib.load("iris_model.joblib")
model = data["model"]
target_names = data["target_names"]

@app.route("/")
def home():
    return """
    <h1>Iris Flower Prediction API</h1>
    <p>Cloud deployment is successful!</p>
    <p>Use POST /predict to predict flower species.</p>
    """

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        if not isinstance(input_data, dict) or "features" not in input_data:
            return jsonify({"error": "Provide features as a JSON array"}), 400

        features = np.array(input_data["features"], dtype=float)

        if features.shape != (4,) or not np.all(np.isfinite(features)):
            return jsonify({"error": "Provide exactly four valid numeric features"}), 400

        prediction = model.predict([features])[0]

        return jsonify({
            "prediction": int(prediction),
            "flower_name": target_names[prediction],
            "status": "success"
        })

    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
