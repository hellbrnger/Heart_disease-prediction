from flask import Flask, render_template, request
import numpy as np
import pickle


with open("model.pkl", "rb") as file:
    model = pickle.load(file)


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
      
        input_features = [float(x) for x in request.form.values()]
        features_array = np.array(input_features).reshape(1, -1)

     
        prediction = model.predict(features_array)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        return render_template("result.html", result=result)

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)
