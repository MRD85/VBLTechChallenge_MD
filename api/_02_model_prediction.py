import os
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import pickle
from api.preprocessing_featurepipeline.py import Preprocessor

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


model = load('./model/random_forest_weights.joblib')



app = FastAPI()

@app.post('/predict')
async def predict(input_data: InputData):
    input_df = pd.DataFrame([input_data.dict()])
    preprocessor = Preprocessor(input_df)
    preprocessed_data = preprocessor.preprocess_data()
    probabilities = model.predict_proba(preprocessed_data)
    
    # Get the probability of class 1 (SeriousDlqin2yrs)
    probability_class_1 = probabilities[:, 1]
    
    response = [{'ID': id, 'Probability': prob} for id, prob in zip(input_df["ID"], probability_class_1)]
    return response
