import os
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, validator, Field
from typing import List
from joblib import load
from _00_preprocessing_featurepipeline import Preprocessor

# Load the trained model
model = load('random_forest_weights.joblib')

# Create a FastAPI app
app = FastAPI()

class InputData(BaseModel):
    ID: int
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float
    MonthlyIncome: int
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: int
    SeriousDlqin2yrs: int

    class Config:
        alias_generator = lambda field_name: field_name.replace("-", "_")
        allow_population_by_field_name = True


    @validator('RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30_59DaysPastDueNotWorse',
               'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
               'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
               'NumberOfTime60_89DaysPastDueNotWorse', 'NumberOfDependents', pre=True)
    def convert_to_number(cls, value, field):
        try:
            if field.type_ is int:
                return int(value)
            elif field.type_ is float:
                return float(value)
        except ValueError:
            raise ValueError(f"{field.alias} must be a valid number.")

@app.post('/predict')
async def predict(input_data: List[InputData]):
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([obj.dict() for obj in input_data])

    # Preprocess the input data
    preprocessor = Preprocessor(input_df)
    preprocessed_data = preprocessor.preprocess_data()

    # Use the trained model to obtain the probabilities
    probabilities = model.predict_proba(preprocessed_data)

    # Extract the probability of the positive class (SeriousDlqin2yrs)
    probabilities_positive_class = probabilities[:, 1]

    # Create the output in the required format
    output = []
    for i, obj in enumerate(input_data):
        output.append({"Id": str(obj.ID), "Probability": probabilities_positive_class[i]})

    # Return the predicted output in JSON format
    return output
