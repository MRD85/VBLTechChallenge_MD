import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from _00_preprocessing_featurepipeline import Preprocessor

class InputData(BaseModel):
    ID: int
    UnsecLines: float
    age: int
    Late3059: int
    DebtRatio: float
    MonthlyIncome: int
    OpenCredit: int
    Late90: int
    PropLines: int
    Late6089: int
    Deps: int

# Load the model using pickle
with open('random_forest_weights.pkl', 'rb') as file:
    model = pickle.load(file)

# Get the expected feature names
expected_features = model.feature_names_in_

app = FastAPI()

@app.post('/predict')
async def predict(input_data: InputData):
    input_df = pd.DataFrame([input_data.dict()])
    
    # Check for missing features and add them to the input_df with appropriate imputed values
    for feature in expected_features:
        if feature not in input_df.columns:
            # Impute missing values. Here we use mean imputation as an example:
            imputed_value = 0  # Replace this with the appropriate imputed value
            input_df[feature] = imputed_value

    preprocessor = Preprocessor(input_df)
    preprocessed_data = preprocessor.preprocess_data()
    probabilities = model.predict_proba(preprocessed_data)
    
    # Get the probability of class 1 (SeriousDlqin2yrs)
    probability_class_1 = probabilities[:, 1]
    
    response = [{'ID': id, 'Probability': prob} for id, prob in zip(input_df["ID"], probability_class_1)]
    return response
