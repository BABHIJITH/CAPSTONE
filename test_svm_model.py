import os
import cv2
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extractor import get_feature_extractor, extract_features

# Load trained model and scaler
svm_model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load genuine and forged data
genuine_folder = 'data/genuine/'
forged_folder = 'data/forged/'

model = get_feature_extractor()

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

genuine_features, genuine_labels = load_data(genuine_folder, label=1)  
forged_features, forged_labels = load_data(forged_folder, label=0)

# Combine features and labels
X = np.array(genuine_features + forged_features)
y = np.array(genuine_labels + forged_labels)

# Scale features
X_scaled = scaler.transform(X)

# Make predictions
predictions = svm_model.predict(X_scaled)

# Performance metrics
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)

# Print metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y, predictions)

# Print Confusion Matrix
print("\nConfusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Forged", "Genuine"], yticklabels=["Forged", "Genuine"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
