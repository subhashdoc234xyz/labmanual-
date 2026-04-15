# Ex.No:6 (b) - Build Random Forest Tree

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import seaborn as sns

# reading the dataset from the local folder
data = pd.read_csv('covid19.csv')

# FIX: Manual uses 3-class Label (0,1,2) based on infection_order
data['Label'] = data['infection_order'].apply(lambda x: 0 if x == 1 else (1 if x == 2 else 2))

print(data.head())

# Unique values
for i in data.columns:
    print(data[i].unique(), "\t", data[i].nunique())

data.info()

for i in data.columns:
    print(data[i].value_counts())
    print()

# ✅ FIX: Use only numeric columns for correlation
fig = plt.figure(figsize=(10, 6))
sns.heatmap(data.select_dtypes(include=[np.number]).corr(), annot=True)
plt.show()

# Encoding
le = LabelEncoder()
for sex in data.columns:
    data[sex] = le.fit_transform(data[sex])
for age in data.columns:
    data[age] = le.fit_transform(data[age])
for country in data.columns:
    data[country] = le.fit_transform(data[country])
for province in data.columns:
    data[province] = le.fit_transform(data[province])
for city in data.columns:
    data[city] = le.fit_transform(data[city])
for infection_case in data.columns:
    data[infection_case] = le.fit_transform(data[infection_case])
for state in data.columns:
    data[state] = le.fit_transform(data[state])

print(data.head())

# Second heatmap
fig = plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True)
plt.show()

# Features and label
X = data[data.columns[:-1]]
y = data['Label']
print(X.head(2))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy for random forest:", metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred));
print("\nThus, the python program for random forest tree was executed successfully.")