from flask import Flask, render_template, request, redirect, url_for
import os
import joblib
from feature_extractor import get_feature_extractor, extract_features
import numpy as np
import cv2

app = Flask(__name__)

# Load the pre-trained SVM model and scaler
svm_model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Initialize the VGG16 model for feature extraction
feature_extractor = get_feature_extractor()

# Path for saving uploaded files
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route for uploading signatures
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file upload and verification
@app.route('/verify', methods=['POST'])
def verify_signature():
    if 'signature' not in request.files:
        return redirect(request.url)
    
    file = request.files['signature']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Extract features from the uploaded image
        features = extract_features(filename, feature_extractor)
        features_scaled = scaler.transform([features])  # Standardize the features
        
        # Make prediction with the SVM model
        prediction = svm_model.predict(features_scaled)
        
        result = "Genuine" if prediction[0] == 1 else "Forged"
        
        return render_template('result.html', result=result, image_path=filename)

if __name__ == '__main__':
    app.run(debug=True)
