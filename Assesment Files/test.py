import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle


train = pd.read_csv("Assesment Files//data//cs-test.csv")
test_set = pd.read_csv("Assesment Files//data//cs-training.csv")
train = train.dropna(subset=['SeriousDlqin2yrs'])


drop_columns = ['Unnamed: 0']
train = train.drop(drop_columns + ['RevolvingUtilizationOfUnsecuredLines',
'NumberOfTime30-59DaysPastDueNotWorse',
'DebtRatio',
'MonthlyIncome',
'NumberOfOpenCreditLinesAndLoans',
'NumberOfTimes90DaysLate',
'NumberRealEstateLoansOrLines',
'NumberOfTime60-89DaysPastDueNotWorse',
'NumberOfDependents'], axis=1)

train = train.rename(columns={
'RevolvingUtilizationOfUnsecuredLines': 'UnsecLines',
'NumberOfTime30-59DaysPastDueNotWorse': 'Late3059',
'DebtRatio': 'DebtRatio',
'MonthlyIncome': 'MonthlyIncome',
'NumberOfOpenCreditLinesAndLoans': 'OpenCredit',
'NumberOfTimes90DaysLate': 'Late90',
'NumberRealEstateLoansOrLines': 'PropLines',
'NumberOfTime60-89DaysPastDueNotWorse': 'Late6089',
'NumberOfDependents': 'Deps'
})

X_train = train
y_train = train['SeriousDlqin2yrs']

pipe = make_pipeline(SimpleImputer(strategy='median'),
StandardScaler())


X_train = pipe.fit_transform(X_train)


parameters = {'n_estimators': [100, 300, 500, 800, 1000],
'max_features': ['sqrt', 'log2'],
'random_state': [42]}
clf = GridSearchCV(RandomForestClassifier(), parameters)
clf.fit(X_train, y_train)

with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(clf, file)