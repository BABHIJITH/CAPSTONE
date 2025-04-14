import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Initialize the VGG16 model without the top layers (for feature extraction)
def get_feature_extractor():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

# Function to extract features from an image using the VGG16 model
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # Preprocess the image as per VGG16
    features = model.predict(x)  # Extract features
    return features.flatten()  # Flatten the features to a 1D array
