import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from feature_extractor import save_extracted_features_and_scaler

# Paths
feature_folder = 'data'  # Folder containing 'genuine' and 'forged'
scaler_path = 'models/scaler.pkl'
svm_model_path = 'models/svm_model.pkl'

# Ensure the model directory exists
os.makedirs(os.path.dirname(svm_model_path), exist_ok=True)

# Step 1: Extract features and scale them
try:
    X_scaled, y = save_extracted_features_and_scaler(feature_folder, scaler_path)
except Exception as e:
    print(f"Error extracting features: {e}")
    exit()

# Step 2: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Train the SVM classifier
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Step 4: Save the trained model
joblib.dump(svm_classifier, svm_model_path)
print(f"SVM model saved to {svm_model_path}")

# Step 5: Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)

print("\n=== Model Evaluation ===")
print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"AUC-ROC Score : {auc_score:.4f}")

# Step 7: Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Forged", "Genuine"], yticklabels=["Forged", "Genuine"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
