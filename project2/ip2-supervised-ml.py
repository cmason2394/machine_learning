# -*- coding: utf-8 -*-
"""
Cassi Mason
Individual Project 2: Supervised Machine Learning
CS379: Machine Learning
12/02/2024
"""
# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 


# *******************************************************
# ingest data 
filepath1 = 'oura_2023-12-01_2024-12-02_trends.csv'
filepath2 = 'oura_2022-01-24_2024-12-04_trends.csv'

# display every column in df when printing to console
pd.set_option('display.max_columns', None)

# import csv as a pandas dataframe
df = pd.read_csv(filepath2)
print(df.head())

# look at the different variables
columns = df.columns.tolist()
print(columns)

# variable types, counts, number of columns
df.info()

# look at columns listed as objects
df_obj = df[['date', 'Bedtime End', 'Bedtime Start']]
print(df_obj.head())

# number of null values
print('Columns with null values before processing:', df.isnull().sum(), sep = '\n')

# *******************************************************

# process data
# convert objects to date or datetime
# create more columns for dates to be month, week, day of week, hour, minute?
# replace null values with mean value for each column
# loop over every column
for col in columns:
    # check if its a numeric column
    if df[col].dtype == np.int64 or df[col].dtype == np.float64:
        # if it is numeric data type
        # determine the mean value for the column
        mean_val = df[col].mean()
        # replace NaN values with the mean value
        df[col] = df[col].fillna(mean_val)

print('Columns with null values after processing:', df.isnull().sum(), sep = '\n')
# *******************************************************

# exploratory analysis
# correlation matrix to see relationships between variables
corr_matrix = df.drop(df_obj, axis=1).corr()
print('how variables are correlated with deep sleep')
print(corr_matrix['Deep Sleep Score'].sort_values())
print('how variables are correlated with HRV balance')
print(corr_matrix['HRV Balance Score'].sort_values())
# *******************************************************

# supervised machine learning
# split data into feature data and target data
X = df.drop('Deep Sleep Score')
y = df['Deep Sleep Score']

# split data into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=333)

# build linear regression model
model = LinearRegression()


# *******************************************************

# results
# run cross validation to get estimate on accuracy and variance of predictions
estimates = cross_val_predict(model, X, y)

# calculate metrics, how accurate and variable model is
mse = mean_squared_error(y, estimates)
r2 = r2_score(y, estimates)

print("Cross-Validation MSE:", mse)
print("Cross-Validation R²:", r2)
# *******************************************************
