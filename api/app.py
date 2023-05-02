import os
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from api.preprocessing_featurepipeline import Preprocessor
from joblib import load

# Load the trained model

model = load('./model/random_forest_weights.joblib')


# Create a FastAPI app
app = FastAPI()

# Define the input data schema
class InputData(BaseModel):
    ID: int
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: int
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: int

@app.post('/predict')
async def predict(input_data: InputData):
    # Replace "-" characters with "_" characters in the input dictionary
    input_dict = {key.replace("-", "_"): value for key, value in input_data.dict().items()}
    
    # Create an InputData instance from the updated dictionary
    updated_input_data = InputData(**input_dict)

    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([updated_input_data.dict()])

    # Preprocess the input data
    preprocessor = Preprocessor(input_df)
    preprocessed_data = preprocessor.preprocess_data()

    # Use the trained model to make a prediction
    prediction = model.predict(preprocessed_data)

    # Extract the probability of the positive class
    probability = prediction[:, 1][0]

    # Return the predicted output in JSON format
    return {'probability': probability}

