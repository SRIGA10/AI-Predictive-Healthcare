from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained models
dt_model = joblib.load("decision_tree.pkl")
lr_model = joblib.load("logistic_regression.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    bmi = float(request.form["bmi"])
    bp = float(request.form["bp"])
    glucose = float(request.form["glucose"])
    heart_rate = float(request.form["heart_rate"])
    cholesterol = float(request.form["cholesterol"])

    data = np.array([
        [age, bmi, bp, glucose, heart_rate, cholesterol]
    ])

    # Model probabilities
    dt_probability = dt_model.predict_proba(data)[0][1]
    lr_probability = lr_model.predict_proba(data)[0][1]

    # Combined risk percentage
    risk_percentage = round(
        ((dt_probability + lr_probability) / 2) * 100,
        1
    )

    # Risk category
    if risk_percentage < 35:
        result = "Low Risk"
        risk_class = "low"

    elif risk_percentage < 65:
        result = "Moderate Risk"
        risk_class = "medium"

    else:
        result = "Higher Risk"
        risk_class = "high"

    # Individual model predictions
    dt_prediction = dt_model.predict(data)[0]
    lr_prediction = lr_model.predict(data)[0]

    # Patient parameter indicators
    age_score = np.clip((age - 18) / 62, 0, 1) * 20
    bmi_score = np.clip((bmi - 18) / 22, 0, 1) * 15
    bp_score = np.clip((bp - 90) / 90, 0, 1) * 25
    glucose_score = np.clip((glucose - 70) / 130, 0, 1) * 25
    heart_score = np.clip((heart_rate - 60) / 80, 0, 1) * 5
    cholesterol_score = np.clip(
        (cholesterol - 120) / 180, 0, 1
    ) * 10

    parameter_scores = [
        {"name": "Age", "value": round(age_score, 1)},
        {"name": "BMI", "value": round(bmi_score, 1)},
        {"name": "Blood Pressure", "value": round(bp_score, 1)},
        {"name": "Glucose", "value": round(glucose_score, 1)},
        {"name": "Heart Rate", "value": round(heart_score, 1)},
        {"name": "Cholesterol", "value": round(cholesterol_score, 1)}
    ]

    return render_template(
        "index.html",
        result=result,
        risk_class=risk_class,
        risk_percentage=risk_percentage,

        dt_result="Higher Risk"
        if dt_prediction
        else "Lower Risk",

        lr_result="Higher Risk"
        if lr_prediction
        else "Lower Risk",

        parameter_scores=parameter_scores
    )


if __name__ == "__main__":
    app.run(debug=True)