import os
import joblib
import numpy as np
from flask import Flask, request, render_template, url_for
from feature_extractor import get_feature_extractor, extract_features

app = Flask(__name__)

# Load the pre-trained SVM model and feature extractor once at startup
svm_model = joblib.load('models/svm_model.pkl')
feature_model = get_feature_extractor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    if 'signature' not in request.files:
        return "No file part", 400
    file = request.files['signature']
    if file.filename == '':
        return "No selected file", 400
    
    # Save the uploaded file in the static/uploads folder
    upload_folder = os.path.join('static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    
    # Extract features from the uploaded signature image
    features = extract_features(file_path, feature_model)
    prediction = svm_model.predict([features])[0]
    probability = svm_model.predict_proba([features])[0]
    
    result = "Genuine Signature" if prediction == 1 else "Forged Signature"
    confidence = max(probability)
    
    return render_template('result.html', result=result, confidence=confidence, filename=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
