import os
import numpy as np
import joblib
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler

# Define the feature extractor function using the VGG16 model
def get_feature_extractor():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    return model

# Extract features from an image and flatten the output
def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        return features.flatten()
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

# Save extracted features and scaler to a file
def save_extracted_features_and_scaler(feature_folder, scaler_path):
    genuine_folder = os.path.join(feature_folder, 'genuine')
    forged_folder = os.path.join(feature_folder, 'forged')

    # Ensure folders exist
    if not os.path.exists(genuine_folder) or not os.path.exists(forged_folder):
        raise FileNotFoundError("One or both image folders are missing!")

    model = get_feature_extractor()

    features = []
    labels = []

    # Extract features from genuine signatures
    for img_name in os.listdir(genuine_folder):
        img_path = os.path.join(genuine_folder, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            feature = extract_features(img_path, model)
            if feature is not None:
                features.append(feature)
                labels.append(1)  # Genuine label

    # Extract features from forged signatures
    for img_name in os.listdir(forged_folder):
        img_path = os.path.join(forged_folder, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            feature = extract_features(img_path, model)
            if feature is not None:
                features.append(feature)
                labels.append(0)  # Forged label

    # Convert features to a NumPy array
    X = np.array(features)
    y = np.array(labels)

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    return X_scaled, y
