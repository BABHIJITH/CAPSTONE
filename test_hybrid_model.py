import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import StandardScaler

# Load the trained SVM model and scaler
svm_model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load genuine and forged data
genuine_folder = 'data/genuine/'
forged_folder = 'data/forged/'

# Initialize ResNet50 model for feature extraction
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_features_cnn(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def load_data(folder, label):
    features = []
    labels = []
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            feature = extract_features_cnn(img_path, resnet_model)
            features.append(feature)
            labels.append(label)
    return features, labels

# Load data
genuine_features, genuine_labels = load_data(genuine_folder, label=1)  
forged_features, forged_labels = load_data(forged_folder, label=0)

# Combine features and labels
X = np.array(genuine_features + forged_features)
y = np.array(genuine_labels + forged_labels)

# Scale the features
X_scaled = scaler.transform(X)

# Make predictions with SVM using features from CNN
predictions = svm_model.predict(X_scaled)

# Performance metrics
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)

# Print performance metrics
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
