
from flask import Flask, render_template, request
from pathlib import Path
import pickle

app = Flask(__name__)

# Get the directory containing this file
BASE_DIR = Path(__file__).resolve().parent

# Load the trained model and scaler
with open(BASE_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read the flower measurements
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Validate the input
        features = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        # Scale the input and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        # Support numeric or string class labels
        flower_names = {
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica",
            "0": "Setosa",
            "1": "Versicolor",
            "2": "Virginica",
            "setosa": "Setosa",
            "versicolor": "Versicolor",
            "virginica": "Virginica",
        }

        result = flower_names.get(
            prediction.item() if hasattr(prediction, "item")
            else prediction
        )

        if result is None:
            result = str(prediction)

        return render_template(
            "index.html",
            prediction=result
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            error="Please enter valid numerical measurements."
        ), 400


# Local development only; Vercel imports the app instance above.
if __name__ == "__main__":
    app.run(debug=True)
