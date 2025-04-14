import os
import joblib
import numpy as np
from flask import Flask, request, render_template
from enhancenet.enhance_utils import enhance_signature  
from cnn_model.extract_features import extract_features 

app = Flask(__name__)


svm_model = joblib.load('models/svm_model.pkl')  
feature_model = get_feature_extractor()  


UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """
    Renders the home page with the upload form for signature verification.
    """
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    """
    Handles the signature verification process: file upload, image enhancement, feature extraction, and prediction.
    """
    if 'signature' not in request.files:
        return "No file part", 400
    
    file = request.files['signature']
    
    if file.filename == '':
        return "No selected file", 400

    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
   
    enhanced_image = enhance_signature(file_path)  
    enhanced_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'enhanced_' + file.filename)
    enhanced_image.save(enhanced_image_path)  
    
    features = extract_features(enhanced_image_path, feature_model)
    

    prediction = svm_model.predict([features])[0]
    probability = svm_model.predict_proba([features])[0]
    
   
    result = "Genuine Signature" if prediction == 1 else "Forged Signature"
    confidence = max(probability)  
    
   
    return render_template('result.html', result=result, confidence=confidence, filename=enhanced_image_path)

if __name__ == '__main__':
    app.run(debug=True)
