from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')  # If you used scaling in your model


def calculate_dpf(parents, siblings, grandparents):
    """Calculate Diabetes Pedigree Function (without uncles/aunts)"""
    contributions = {
        'parent': {'risk': 0.5, 'weight': 1.0},
        'sibling': {'risk': 0.3, 'weight': 0.8},
        'grandparent': {'risk': 0.2, 'weight': 0.6}
    }

    parent_contrib = parents * contributions['parent']['risk'] * contributions['parent']['weight']
    sibling_contrib = siblings * contributions['sibling']['risk'] * contributions['sibling']['weight']
    grandparent_contrib = grandparents * contributions['grandparent']['risk'] * contributions['grandparent']['weight']

    dpf = parent_contrib + sibling_contrib + grandparent_contrib
    return min(dpf, 2.5)  # Hard cap at 2.5


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        pregnancy = float(request.form['Pregnancy'])
        glucose = float(request.form['Glucose'])
        blood_pressure = float(request.form['BloodPressure'])
        skin_thickness = float(request.form['SkinThickness'])
        insulin = float(request.form['Insulin'])
        height = float(request.form['Height'])  # Height in meters
        weight = float(request.form['Weight'])  # Weight in kg
        age = float(request.form['Age'])

        # Get family history data
        diabetic_parents = int(request.form['DiabeticParents'])
        diabetic_siblings = int(request.form['DiabeticSiblings'])
        diabetic_grandparents = int(request.form['DiabeticGrandparents'])

        # Calculate DPF from family history
        dpf = calculate_dpf(
            parents=diabetic_parents,
            siblings=diabetic_siblings,
            grandparents=diabetic_grandparents
        )
        print(dpf)

        # Calculate BMI
        bmi = weight / (height ** 2)
        print(f"Calculated BMI: {bmi:.2f}")

        # Prepare features for prediction (ensure order matches model expectations)
        features = np.array([pregnancy,glucose,blood_pressure,skin_thickness,insulin,bmi,dpf,age]).reshape(1, -1)

        # Scale the features (if required by your model)
        scaled_features = scaler.transform(features)
        print("Scaled features:", scaled_features)

        # Make prediction using the model
        prediction = model.predict(scaled_features)
        print("Prediction output:", prediction)

        # Interpret the result
        result = "You are at risk of having diabetes" if prediction[
                                                             0] == 1 else "You are not at risk of having diabetes"
        result_class = "diabetic" if prediction[0] == 1 else "non-diabetic"

        return render_template('result.html', prediction_text=result, result_class=result_class)

    except Exception as e:
        return render_template('result.html', prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)