
import pickle
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# initialising flask application
app = Flask(__name__)

# loading the model
with open("model/model.sav", "rb") as file:
    model = pickle.load(file)

# loading the scaler
with open("model/scaler.sav", "rb") as file:
    scaler = pickle.load(file)


@app.route("/")
def home():
    return "diabetes prediction application is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        data = request.get_json()
        input_data = pd.DataFrame(data)

        if not data:
            return jsonify({"error": "Input data not providrd"}), 400
        
        # validate input
        required_columns = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]

        if not all(column in input_data.columns for column in required_columns):
            return jsonify({"error": f"required columns missing. required columns {required_columns}"}), 400
        
        # scaling the inputs
        scaled_data = scaler.transform(input_data)

        # predictions
        prediction = model.predict(scaled_data)

        # response
        response = {
            "prediction": "diabetic" if prediction[0] == 1 else "non diabetic"
        }


    except:
        pass