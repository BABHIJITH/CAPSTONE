import os
from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
from feature_extractor import get_feature_extractor, extract_features

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
MY_SIGNATURES_FOLDER = 'my_signatures/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

svm_model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_extractor = get_feature_extractor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify_signature():
    if 'signature' not in request.files:
        return redirect(request.url)

    file = request.files['signature']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        features = extract_features(filename, feature_extractor)
        features_scaled = scaler.transform([features])
        prediction = svm_model.predict(features_scaled)

        result = "Genuine" if prediction[0] == 1 else "Forged"

        return render_template('result.html', result=result, image_path=filename)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
