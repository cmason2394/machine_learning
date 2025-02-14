# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:51:25 2024

@author: cassi
"""

#%matplotlib inline 
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex (Female, Male. After processing, Female = 0 and Male = 1)
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''
# display every column in df when printing to console
pd.set_option('display.max_columns', None)

df = pd.read_excel('CS379T-Week-1-IP.xls')
print('before transforming dataset')
print(df.head())
print('Columns with null values before transformations:', df.isnull().sum(), sep = '\n')
df.drop(labels=['body','name'], axis=1, inplace=True)
df.info()
#df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
print(df.head())
df.info()
print('Columns with null values after fill NaN with 0:', df.isnull().sum(), sep = '\n')

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
print('after transforming dataset to all numeric')
print(df.head())
df.info()

# bar plot of survivors
fig = plt.figure(figsize = (10, 5))
categories = ['Died', 'Survived']
counts = df.survived.value_counts()

# creating the bar plot
plt.bar(categories, counts, color ='maroon', 
        width = 0.4)

plt.xlabel("Outcome")
plt.ylabel("No. of passengers")
plt.title("Survival status of the passengers on the Titanic")
plt.show()

# correlation matrix between columns, most interested in correlations of all columns to survival column
corr_matrix = df.corr()
print('correlation matrix between the survival column and every other column')
print(corr_matrix['survived'])

# Process data for unstructured clustering
#df.drop(labels = ['sex', 'boat'], axis=1, inplace=True) # potential columns to drop: 'sex' (more men died than women), 'boat' (lifeboats) 
X = np.array(df.drop(labels = ['survived'], axis=1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

# unstructured clustering with K Means
clf = KMeans(n_clusters=2)
clf.fit(X)

# Measure accuracy of model
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print('accuracy of predicting survival of passengers on titanic:')
print(correct/len(X))

# supervised learning
model = LinearRegression()

# run cross validation to get estimate on accuracy and variance of predictions
estimates = cross_val_predict(model, X, y)

# calculate metrics, how accurate and variable model is
mse = mean_squared_error(y, estimates)
r2 = r2_score(y, estimates)

print("Cross-Validation MSE:", mse)
print("Cross-Validation R²:", r2)

'''
# from https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
scores = cross_val_score(model, X, y, cv=10, scoring = "accuracy")
print('Scores: ', scores)
print('Mean: ', scores.mean())
print('Standard Deviation: ', scores.std())
'''

# Fit the model with the entire dataset
model.fit(X, y)
print("Model trained on entire dataset.")

# visual of linear regression equation
