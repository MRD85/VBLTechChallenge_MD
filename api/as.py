from joblib import load

model = load('./Assesment Files/random_forest_weights.joblib')
print(type(model))

if isinstance(model, pd.DataFrame):
    print("Shape:", model.shape)
    print("Columns:", model.columns)
else:
    print("The loaded data is not a Pandas DataFrame.")