from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import csv
import traceback
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Get the absolute path to the datasets directory
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')

# Load training dataset
try:
    training = pd.read_csv(os.path.join(DATASET_DIR, 'hc training dataset (1).csv'))
    symptoms = training.columns[:-1]  # All columns except the last one
    print("Available symptoms:", list(symptoms))
except Exception as e:
    print("Error loading training dataset:", e)
    symptoms = []

# Train machine learning models
try:
    x = training[symptoms]
    y = training['prognosis']
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    # Train classifiers
    clf = DecisionTreeClassifier().fit(x, y)
    model = SVC().fit(x, y)  # Use Support Vector Classifier as default
except Exception as e:
    print("Error training model:", e)

# Load supplementary datasets
severity_dict = {}
description_dict = {}
precaution_dict = {}

try:
    # Load severity data
    with open(os.path.join(DATASET_DIR, 'healthcare severity dataset (1).csv')) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            severity_dict[row[0]] = int(row[1])

    # Load descriptions
    with open(os.path.join(DATASET_DIR, 'healthcare description dataset (1).csv')) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            description_dict[row[0]] = row[1]

    # Load precautions
    with open(os.path.join(DATASET_DIR, 'healthcare precautions_1 (1).csv')) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            precaution_dict[row[0]] = row[1:]
except Exception as e:
    print("Error loading supplementary datasets:", e)

def validate_symptoms(input_symptoms):
    """Validate input symptoms and return invalid ones."""
    invalid_symptoms = []
    valid_symptoms = []
    
    for symptom in input_symptoms:
        if symptom not in symptoms:
            invalid_symptoms.append(symptom)
        else:
            valid_symptoms.append(symptom)
    
    return valid_symptoms, invalid_symptoms

# API to predict disease
@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        # Parse input data
        data = request.get_json()
        symptoms_present = data.get('symptoms', [])
        
        days = data.get('days', 1)

        if not symptoms_present:
            return jsonify({
                'error': "Please enter at least one symptom."
            }), 400

        # Validate symptoms
        valid_symptoms, invalid_symptoms = validate_symptoms(symptoms_present)
        
        if not valid_symptoms:
            error_msg = "Invalid symptoms entered: " + ", ".join(invalid_symptoms)
            if len(symptoms) > 0:
                error_msg += "\nExample valid symptoms: " + ", ".join(list(symptoms)[:5])
            return jsonify({'error': error_msg}), 400

        if invalid_symptoms:
            print(f"Warning: Ignoring invalid symptoms: {invalid_symptoms}")

        # Create input vector for the model
        x_input = [1 if symptom in valid_symptoms else 0 for symptom in symptoms]
        prediction = model.predict([x_input])

        # Decode prediction
        disease = le.inverse_transform(prediction)[0]
        description = description_dict.get(disease, "No description available")
        precautions = precaution_dict.get(disease, ["No precautions available"])
        severity = "High" if severity_dict.get(disease, 0) > 7 else "Medium" if severity_dict.get(disease, 0) > 4 else "Low"

        # Respond with prediction details
        return jsonify({
            'disease': disease,
            'description': description,
            'precautions': precautions,
            'severity': severity,
            'ignored_symptoms': invalid_symptoms if invalid_symptoms else None
        })
    
    except Exception as e:
        print("Error during prediction:", e)
        print(traceback.format_exc())
        return jsonify({
            'error': "An error occurred during prediction. Please try again."
        }), 500

# API to get available symptoms
@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({
        'symptoms': list(symptoms)
    })

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Service is running'
    })

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # In production, host should be '0.0.0.0' to accept connections from any IP
    host = '0.0.0.0'
    app.run(host=host, port=port, debug=False)
    
