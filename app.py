from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
loaded_model = joblib.load(BASE_DIR / "iris_model.joblib")

if isinstance(loaded_model, dict):
    model = loaded_model["model"]
    target_names = loaded_model.get(
        "target_names",
        ["setosa", "versicolor", "virginica"]
    )
else:
    model = loaded_model
    target_names = ["setosa", "versicolor", "virginica"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            features = [
                float(request.form["sepal_length"]),
                float(request.form["sepal_width"]),
                float(request.form["petal_length"]),
                float(request.form["petal_width"])
            ]

            if not np.all(np.isfinite(features)):
                raise ValueError("Please enter valid measurements.")

            predicted_class = int(model.predict([features])[0])
            prediction = str(target_names[predicted_class])

        except (ValueError, KeyError, TypeError):
            error = "Please enter valid numeric measurements."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "message": "Iris prediction API is running"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or "features" not in data:
        return jsonify({
            "error": "Send JSON with a 'features' list containing four measurements."
        }), 400

    try:
        features = [float(value) for value in data["features"]]

        if len(features) != 4 or not np.all(np.isfinite(features)):
            raise ValueError

        predicted_class = int(model.predict([features])[0])

        return jsonify({
            "prediction": predicted_class,
            "flower_name": str(target_names[predicted_class]),
            "status": "success"
        }), 200

    except (ValueError, TypeError, IndexError):
        return jsonify({
            "error": "Features must contain four valid numeric measurements."
        }), 400


if __name__ == "__main__":
    app.run(debug=False)
