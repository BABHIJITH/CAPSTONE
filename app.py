import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
from werkzeug.utils import secure_filename
from feature_extractor import get_feature_extractor, extract_features
from datetime import timedelta

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.permanent_session_lifetime = timedelta(minutes=30)

UPLOAD_FOLDER = 'static/uploads/'
REGISTERED_FOLDER = 'static/registered/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REGISTERED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REGISTERED_FOLDER'] = REGISTERED_FOLDER

# Load pre-trained SVM model and scaler
svm_model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_extractor = get_feature_extractor()

# ---------------- Routes ---------------- #

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username == "admin" and password == "password":
        session['user'] = username
        return redirect(url_for('dashboard'))
    else:
        flash("Invalid Credentials!", "danger")
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('index'))
    return render_template('dashboard.html')
@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user' not in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        name = request.form.get('name')
        unique_id = request.form.get('unique_id')
        file = request.files['signature']

        if name and unique_id and file:
            user_folder = os.path.join(app.config['REGISTERED_FOLDER'], unique_id)
            os.makedirs(user_folder, exist_ok=True)

            count = len(os.listdir(user_folder)) + 1
            filename = secure_filename(f"{unique_id}_{count}.jpg")
            filepath = os.path.join(user_folder, filename)
            file.save(filepath)

            flash("Signature registered successfully!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("All fields are required!", "danger")

    return render_template('register.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if 'user' not in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        identifier = request.form.get('identifier')
        captured_file = request.files['signature']

        if identifier and captured_file:
            user_folder = os.path.join(app.config['REGISTERED_FOLDER'], identifier)
            if not os.path.exists(user_folder):
                flash("No registered signature found for this ID.", "danger")
                return redirect(url_for('verify'))

            captured_filename = secure_filename(captured_file.filename)
            captured_path = os.path.join(app.config['UPLOAD_FOLDER'], captured_filename)
            captured_file.save(captured_path)

            features_captured = extract_features(captured_path, feature_extractor)
            features_captured_scaled = scaler.transform([features_captured])

            result = "Forged"
            for file_name in os.listdir(user_folder):
                registered_path = os.path.join(user_folder, file_name)
                features_registered = extract_features(registered_path, feature_extractor)
                features_registered_scaled = scaler.transform([features_registered])
                prediction = svm_model.predict(features_captured_scaled)
                if prediction[0] == 1:
                    result = "Genuine"
                    break

            return render_template('result.html', result=result, image_filename=captured_filename)

        flash("All fields are required!", "danger")
        return redirect(url_for('verify'))

    return render_template('verify.html')


if __name__ == '__main__':
    app.run(debug=True)
