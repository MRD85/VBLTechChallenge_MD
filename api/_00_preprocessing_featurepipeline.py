import pandas as pd
import numpy as np

class Preprocessor:

    def __init__(self, data):
        self.data = data

    def preprocess_data(self):
        self._rename_columns()
        self._impute_missing_values()
        self._cap_outliers()
        self._one_hot_encoding()
        self._remove_extra_features()
        return self.data

    def _rename_columns(self):
        self.data = self.data.rename(columns={
            'RevolvingUtilizationOfUnsecuredLines': 'UnsecLines',
            'NumberOfTime30_59DaysPastDueNotWorse': 'Late3059',
            'DebtRatio': 'DebtRatio',
            'MonthlyIncome': 'MonthlyIncome',
            'NumberOfOpenCreditLinesAndLoans': 'OpenCredit',
            'NumberOfTimes90DaysLate': 'Late90',
            'NumberRealEstateLoansOrLines': 'PropLines',
            'NumberOfTime60_89DaysPastDueNotWorse': 'Late6089',
            'NumberOfDependents': 'Deps',
            'SeriousDlqin2yrs': 'Target'
        })

    def _impute_missing_values(self):
        self.data.MonthlyIncome = self.data.MonthlyIncome.fillna(self.data.MonthlyIncome.median())
        self.data.Deps = self.data.Deps.fillna(self.data.Deps.median())

    def _cap_outliers(self):
        self.data.loc[self.data.Late90 >= 5, 'Late90'] = 5
        self.data.loc[self.data.PropLines >= 6, 'PropLines'] = 6
        self.data.loc[self.data.Late6089 >= 3, 'Late6089'] = 3
        self.data.loc[self.data.Deps >= 4, 'Deps'] = 4

    def _one_hot_encoding(self):
        self.data = pd.get_dummies(self.data, columns=["UnsecLines"], prefix="UnsecLines")
        self.data = pd.get_dummies(self.data, columns=["OpenCredit"], prefix="OpenCredit")
        self.data = pd.get_dummies(self.data, columns=["Late90"], prefix="Late90")
        self.data = pd.get_dummies(self.data, columns=["PropLines"], prefix="PropLines")
        self.data = pd.get_dummies(self.data, columns=["Late6089"], prefix="Late6089")
        self.data = pd.get_dummies(self.data, columns=["Deps"], prefix="Deps")

    def _remove_extra_features(self):
        features_to_remove = set(self.data.columns) - set([
            'ID',
            'UnsecLines',
            'age',
            'Late3059',
            'DebtRatio',
            'MonthlyIncome',
            'OpenCredit',
            'Late90',
            'PropLines',
            'Late6089',
            'Deps'
        ])
        self.data = self.data.drop(labels=features_to_remove, axis=1)
