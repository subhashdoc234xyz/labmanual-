# EXERCISE 6(a) - DECISION TREE CLASSIFIER
# EXACT CODE AS PER MANUAL PAGES 35-43

# import Python library packages
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

# ================== FIXES (NO LOGIC CHANGE) ==================
# Handle missing values
data['infection_order'] = data['infection_order'].fillna(0)

# Create Label column if not present
if 'Label' not in data.columns:
    data['Label'] = data['infection_order'].apply(
        lambda x: 0 if x == 1 else (1 if x == 2 else 2)
    )
# ============================================================

# In[2]:
print("First 5 rows of data:")
print(data.head())

# In[3]:
print("\nUnique values in each column:")
for i in data.columns:
    print(data[i].unique(), "\t", data[i].nunique())

# In[4]:
print("\nData info:")
print(data.info())

# In[5]:
print("\nValue counts for each column:")
for i in data.columns:
    print(data[i].value_counts())
    print()

# In[6]:
print("\nGenerating correlation heatmap...")
fig = plt.figure(figsize=(10,6))
numeric_cols = data.select_dtypes(include=[np.number]).columns
sns.heatmap(data[numeric_cols].corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()

# In[7]:
print("\nConverting string columns to integers...")
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object' or data[col].dtype == 'str':
        data[col] = le.fit_transform(data[col].astype(str))

print("\nData after encoding (first 5 rows):")
print(data.head())

# In[8]:
fig = plt.figure(figsize=(10,6))
numeric_cols = data.select_dtypes(include=[np.number]).columns
sns.heatmap(data[numeric_cols].corr(), annot=True)
plt.title('Correlation Heatmap After Encoding')
plt.show()

# In[9]:
X = data.drop('Label', axis='columns')
y = data['Label']
print("\nFeatures (X) first 2 rows:")
print(X.head(2))

# In[10]:
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn import metrics

print("\n" + "="*60)
print("DECISION TREE RESULTS")
print("="*60)
print("Accuracy for Decision Tree:", metrics.accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nCross Validation Scores:")
print(cross_val_score(model, X, y, cv=10))

print("\n" + "="*60)
print("Result: Thus, the python program for decision tree was executed successfully.")
print("="*60)