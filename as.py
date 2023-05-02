import joblib
import numpy as np

# Load the model from the pickle file
with open('Assesment Files/random_forest_model.pkl', 'rb') as f:
    model = joblib.load(f)

# Create a numpy array from the input data
input_data = np.array([[1, 0.766126609, 45, 2, 0.802982129, 9120, 13, 0, 6, 0, 2]])

# Make a prediction using the loaded model
prediction = model.predict(input_data)

# Print the prediction
print(prediction)