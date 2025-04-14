import os
import cv2
import numpy as np
import joblib
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from feature_extractor import get_feature_extractor, extract_features

# Path to the data folders
genuine_folder = 'data/genuine/'
forged_folder = 'data/forged/'

# Initialize the feature extraction model (VGG16 without the top layers)
model = get_feature_extractor()

# Function to load images and extract features
def load_data(folder, label):
    features = []
    labels = []
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            feature = extract_features(img_path, model)
            features.append(feature)
            labels.append(label)
    return features, labels

# Load data from both genuine and forged folders
genuine_features, genuine_labels = load_data(genuine_folder, label=1)  # Genuine signatures as class 1
forged_features, forged_labels = load_data(forged_folder, label=0)    # Forged signatures as class 0

# Combine the genuine and forged data
X = np.array(genuine_features + forged_features)
y = np.array(genuine_labels + forged_labels)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train an SVM classifier
svm_classifier = svm.SVC(kernel='linear', probability=True)
svm_classifier.fit(X_scaled, y)

# Save the trained model and scaler
joblib.dump(svm_classifier, 'models/svm_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("SVM model and scaler saved!")
