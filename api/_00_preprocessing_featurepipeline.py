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
        self.addMissingColumns()
        self.removeExtraColumns()
        return self.data

    def _rename_columns(self):
        self.data = self.data.rename(columns={
            'ID': 'ID',
            'RevolvingUtilizationOfUnsecuredLines': 'UnsecLines',
            'age': 'age',
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
        self.data.age = pd.qcut(self.data.age.values, 5, duplicates='drop').codes
        self.data.UnsecLines = pd.qcut(self.data.UnsecLines.values, 5, duplicates='drop').codes
        self.data.DebtRatio = pd.qcut(self.data.DebtRatio.values, 5, duplicates='drop').codes
        self.data.MonthlyIncome = pd.qcut(self.data.MonthlyIncome.values, 5, duplicates='drop').codes
        self.data.OpenCredit = pd.qcut(self.data.OpenCredit.values, 5, duplicates='drop').codes
        
        self.data = pd.get_dummies(self.data, columns=["UnsecLines"], prefix="UnsecLines")
        self.data = pd.get_dummies(self.data, columns=["age"], prefix="age")
        self.data = pd.get_dummies(self.data, columns=["Late3059"], prefix="Late3059")
        self.data = pd.get_dummies(self.data, columns=["DebtRatio"], prefix="DebtRatio")
        self.data = pd.get_dummies(self.data, columns=["MonthlyIncome"], prefix="MonthlyIncome")
        self.data = pd.get_dummies(self.data, columns=["OpenCredit"], prefix="OpenCredit")
        self.data = pd.get_dummies(self.data, columns=["Late90"], prefix="Late90")
        self.data = pd.get_dummies(self.data, columns=["PropLines"], prefix="PropLines")
        self.data = pd.get_dummies(self.data, columns=["Late6089"], prefix="Late6089")
        self.data = pd.get_dummies(self.data, columns=["Deps"], prefix="Deps")


    def addMissingColumns(self):
        """ adding columns which are not present in the test data """
        modelFeatures = ['UnsecLines_0', 'UnsecLines_1', 'UnsecLines_2',
                    'UnsecLines_3', 'UnsecLines_4', 'age_0', 'age_1', 'age_2', 'age_3',
                    'age_4', 'Late3059_0', 'Late3059_1', 'Late3059_2', 'Late3059_3',
                    'Late3059_4', 'Late3059_5', 'Late3059_6', 'DebtRatio_0', 'DebtRatio_1',
                    'DebtRatio_2', 'DebtRatio_3', 'DebtRatio_4', 'MonthlyIncome_0',
                    'MonthlyIncome_1', 'MonthlyIncome_2', 'MonthlyIncome_3',
                    'MonthlyIncome_4', 'OpenCredit_0', 'OpenCredit_1', 'OpenCredit_2',
                    'OpenCredit_3', 'OpenCredit_4', 'Late90_0', 'Late90_1', 'Late90_2',
                    'Late90_3', 'Late90_4', 'Late90_5', 'PropLines_0', 'PropLines_1',
                    'PropLines_2', 'PropLines_3', 'PropLines_4', 'PropLines_5',
                    'PropLines_6', 'Late6089_0', 'Late6089_1', 'Late6089_2', 'Late6089_3',
                    'Deps_0.0', 'Deps_1.0', 'Deps_2.0', 'Deps_3.0', 'Deps_4.0']
        for feature in modelFeatures:
            if feature not in self.data.columns:
                self.data[feature] = 0

    def removeExtraColumns(self):
        """ removing columns which are not in the modelFeatures """
        modelFeatures = ['UnsecLines_0', 'UnsecLines_1', 'UnsecLines_2',
                    'UnsecLines_3', 'UnsecLines_4', 'age_0', 'age_1', 'age_2', 'age_3',
                    'age_4', 'Late3059_0', 'Late3059_1', 'Late3059_2', 'Late3059_3',
                    'Late3059_4', 'Late3059_5', 'Late3059_6', 'DebtRatio_0', 'DebtRatio_1',
                    'DebtRatio_2', 'DebtRatio_3', 'DebtRatio_4', 'MonthlyIncome_0',
                    'MonthlyIncome_1', 'MonthlyIncome_2', 'MonthlyIncome_3',
                    'MonthlyIncome_4', 'OpenCredit_0', 'OpenCredit_1', 'OpenCredit_2',
                    'OpenCredit_3', 'OpenCredit_4', 'Late90_0', 'Late90_1', 'Late90_2',
                    'Late90_3', 'Late90_4', 'Late90_5', 'PropLines_0', 'PropLines_1',
                    'PropLines_2', 'PropLines_3', 'PropLines_4', 'PropLines_5',
                    'PropLines_6', 'Late6089_0', 'Late6089_1', 'Late6089_2', 'Late6089_3',
                    'Deps_0.0', 'Deps_1.0', 'Deps_2.0', 'Deps_3.0', 'Deps_4.0']
        extra_columns = [col for col in self.data.columns if col not in modelFeatures]
        self.data.drop(extra_columns, axis=1, inplace=True)