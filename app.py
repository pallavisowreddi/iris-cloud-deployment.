
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "iris_model.joblib"
)
model = joblib.load(MODEL_PATH)

species = ["Setosa", "Versicolor", "Virginica"]

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

            if any(value <= 0 for value in features):
                raise ValueError

            result = model.predict([features])[0]
            prediction = species[result]

        except (ValueError, KeyError):
            error = "Please enter valid positive numeric values."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )
