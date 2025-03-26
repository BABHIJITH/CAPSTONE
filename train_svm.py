import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from feature_extractor import get_feature_extractor, extract_features

def load_dataset(data_dir):
    X, y = [], []
    classes = {'genuine': 1, 'forged': 0}
    extractor = get_feature_extractor()
    
    for label, class_val in classes.items():
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            try:
                feat = extract_features(file_path, extractor)
                X.append(feat)
                y.append(class_val)
            except Exception as e:
                print(f'Error processing {file_path}: {e}')
    return np.array(X), np.array(y)

if __name__ == '__main__':
    data_dir = 'data'
    X, y = load_dataset(data_dir)
    print(f"Loaded {len(X)} samples")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)
    
    print("SVM training complete")
    accuracy = svm.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Save the trained model
    os.makedirs('models', exist_ok=True)
    joblib.dump(svm, 'models/svm_model.pkl')
    print("Trained model saved as models/svm_model.pkl")
