import pickle
import numpy as np

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Note: You can also use explicit imports like 'from sklearn.model_selection import train_test_split'

# Config
DATA_FILE = 'data.pickle'
MODEL_OUTPUT_FILE = 'model.pkl' # Standard extension for pickle model files
TEST_SIZE = 0.2 # 20% of data for testing

print(f"Loading features and labels from {DATA_FILE}...")

try:
    # Use 'with open' for safer, automatic file closing
    with open(DATA_FILE, 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Required file '{DATA_FILE}' not found. Run the feature extraction script first.")
    exit()

# Extract and convert data/labels to NumPy arrays
X = np.asarray(data_dict['data'])
Y = np.asarray(data_dict['labels'])

print(f"Total samples loaded: {len(X)}")

# Stratified splitting ensures an equal representation of classes in training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, 
    Y, 
    test_size=TEST_SIZE, 
    shuffle=True, 
    stratify=Y # Crucial for balanced class representation
)

print(f"Data split: Training size = {len(X_train)}, Testing size = {len(X_test)}")
print("Training Random Forest Classifier...")
# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42) # Added a random_state for reproducibility

# Fit the model to the training data
model.fit(X_train, Y_train)

print("Evaluating model performance on test set...")
# Make predictions on the unseen test data
Y_predict = model.predict(X_test)

# Calculate the accuracy score
score = accuracy_score(Y_predict, Y_test)

# Print the result using an f-string
print(f"\n✅ Classification Complete!")
print(f"Accuracy Score: {score * 100:.2f}% of samples were classified correctly.")

# Save the trained model to a file
print(f"\nSaving trained model to {MODEL_OUTPUT_FILE}...")
with open(MODEL_OUTPUT_FILE, 'wb') as f:
    # Save the model object directly for later loading
    pickle.dump(model, f) 
    
print("Model successfully saved.")