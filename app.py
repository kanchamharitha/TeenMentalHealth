from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
from joblib import dump
from joblib import load
model = load("model.pkl")

dump(model, "model.pkl")

\

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    social_media = float(request.form['social_media'])

    # New fields
    platform = int(request.form['platform'])
    social_interaction = int(request.form['social_interaction'])

    sleep = float(request.form['sleep'])
    screen = float(request.form['screen'])
    academic = float(request.form['academic'])
    physical = float(request.form['physical'])
    stress = int(request.form['stress'])
    anxiety = int(request.form['anxiety'])
    addiction = int(request.form['addiction'])

    data = np.array([[
        age,
        gender,
        social_media,
        platform,
        sleep,
        screen,
        academic,
        physical,
        social_interaction,
        stress,
        anxiety,
        addiction
    ]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "High Risk of Depression"
    else:
        result = "Low Risk of Depression"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)