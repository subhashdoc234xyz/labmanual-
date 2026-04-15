# EXERCISE 5(b) - LOGISTIC REGRESSION MODEL
# EXACT CODE AS PER MANUAL PAGES 30-34 - CORRECTED FOR PYTHON SCRIPT

import pandas as pd
from matplotlib import pyplot as plt

# In[16]:
df = pd.read_csv("insurance_data.csv")
print("First 5 rows of data:")
print(df.head())

# Out[16]:
#    age  bought_insurance
# 0   22                 0
# 1   25                 0
# 2   47                 1
# 3   52                 0
# 4   46                 1

# In[17]:
plt.scatter(df.age, df.bought_insurance, marker='+', color='red')
plt.xlabel('Age')
plt.ylabel('Bought Insurance')
plt.title('Insurance Data')
plt.show()

# Out[17]:
# <matplotlib.collections.PathCollection at 0x20a8cb15d30>

# In[18]:
from sklearn.model_selection import train_test_split

# In[29]:
X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, train_size=0.8, random_state=42)

# In[30]:
print("\nTest data:")
print(X_test)

# Out[30]:
#     age
# 4    46
# 8    62
# 26   23
# 17   58
# 24   50
# 25   54

# In[31]:
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)

# In[66]:
model.fit(X_train, y_train)

# Out[66]:
# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, max_iter=100, multi_class='warn',
#                    n_jobs=None, penalty='l2', random_state=None, solver='warn',
#                    tol=0.0001, verbose=0, warm_start=False)

# In[9]:
print("\nTest data:")
print(X_test)

# Out[9]:
#     age
# 16   25
# 21   26
# 2    47

# In[39]:
y_predicted = model.predict(X_test)
print("\nPredictions:", y_predicted)

# In[33]:
print("\nPrediction probabilities:")
print(model.predict_proba(X_test))

# Out[33]:
# array([[0.40569485, 0.59430515],
#        [0.26002994, 0.73997006],
#        [0.63939494, 0.36060506],
#        [0.29321765, 0.70678235],
#        [0.36637568, 0.63362432],
#        [0.32875922, 0.67124078]])

# In[34]:
print("\nModel Score:", model.score(X_test, y_test))

# Out[34]:
# 1.0

# In[40]:
print("\nPredictions:", y_predicted)

# Out[40]:
# array([1, 1, 0, 1, 1, 1], dtype=int64)

# In[37]:
print("\nTest data:")
print(X_test)

# Out[37]:
#     age
# 4    46
# 8    62
# 26   23
# 17   58
# 24   50
# 25   54

# model.coef_ indicates value of m in y=m*x + b equation
# In[67]:
print("\nCoefficient:", model.coef_)

# Out[67]:
# array([[0.04150133]])

# model.intercept_ indicates value of b in y=m*x + b equation
# In[68]:
print("Intercept:", model.intercept_)

# Out[68]:
# array([-1.52726963])

# Lets defined sigmoid function now and do the math with hand
# In[43]:
import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# In[75]:
def prediction_function(age):
    z = 0.042 * age - 1.53  # 0.04150133 ~ 0.042 and -1.52726963 ~ -1.53
    y = sigmoid(z)
    return y

# In[76]:
age = 35
print("\nPrediction for age 35:", prediction_function(age))

# Out[76]:
# 0.4850044983805899

print("0.485 is less than 0.5 which means person with 35 age will not buy insurance")

# In[77]:
age = 43
print("Prediction for age 43:", prediction_function(age))

# Out[77]:
# 0.568565299077705

print("0.568 is more than 0.5 which means person with 43 will buy the insurance")

print("\n" + "="*60)
print("Result: Thus, the python program for logistics regression model was executed successfully.")
print("="*60),