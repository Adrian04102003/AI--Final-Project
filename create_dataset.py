import os
import pickle
import warnings
from pathlib import Path

import cv2
import mediapipe as mp

# Suppress warnings that clutter the console
warnings.filterwarnings("ignore", category=UserWarning)

# Config
DATA_DIR = Path('./data')
# 21 landmarks per hand * 2 coordinates (x, y) = 42 features
EXPECTED_FEATURES = 42 
MODEL_OUTPUT_FILE = 'data.pickle'

# Initialize the Hand tracking model
mp_hands = mp.solutions.hands
# Set static_image_mode=True as we are processing static images from the dataset
hands_model = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

feature_vectors = []
class_labels = []

print(f"Starting feature extraction from directory: {DATA_DIR}")

# Iterate through each class directory (e.g., '0', '1', '2'...)
for class_dir_name in os.listdir(DATA_DIR):
    class_path = DATA_DIR / class_dir_name
    
    # Skip non-directories or hidden files
    if not class_path.is_dir():
        continue
    
    # Iterate through every image in the current class directory
    for image_file_name in os.listdir(class_path):
        image_path = class_path / image_file_name
        
        current_features = []
        
        # Load and convert image to RGB for Mediapipe
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read image file {image_file_name}. Skipping.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run detection
        results = hands_model.process(img_rgb)
        
        if results.multi_hand_landmarks:
            # It assume one hand per image for simple classification
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Collect all X and Y coordinates to find the bounding box
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]

            min_x = min(x_coords)
            min_y = min(y_coords)
            
            # This makes the features scale- and position-invariant.
            for x, y in zip(x_coords, y_coords):
                current_features.append(x - min_x) # Relative X coordinate
                current_features.append(y - min_y) # Relative Y coordinate

            # Validiation and Storage
            if len(current_features) == EXPECTED_FEATURES:
                feature_vectors.append(current_features)
                class_labels.append(class_dir_name)
            else:
                # This could happen if detection only finds partial landmarks
                print(f"Skipped image {image_file_name} in class {class_dir_name}: Expected {EXPECTED_FEATURES} features but got {len(current_features)}.")

dataset = {'data': feature_vectors, 'labels': class_labels}

with open(MODEL_OUTPUT_FILE, 'wb') as file:
    pickle.dump(dataset, file)

print("\nSummary")
print(f"Feature extraction complete.")
print(f"Total processed samples: {len(feature_vectors)}")
print(f"Dataset saved to: {MODEL_OUTPUT_FILE}")