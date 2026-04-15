# EXERCISE 5(a) - LINEAR REGRESSION MODEL
# EXACT CODE AS PER MANUAL PAGES 27-29

import pandas as pd
import numpy as np
from sklearn import linear_model

# In[2]:
df = pd.read_csv('homeprices.csv')
print("Original data:")
print(df)

# Out[2]:
#    area  bedrooms  age   price
# 0  2600       3.0   20  550000
# 1  3000       4.0   15  565000
# 2  3200       NaN   18  610000
# 3  3600       3.0   30  595000
# 4  4000       5.0    8  760000
# 5  4100       6.0    8  810000

# Data Preprocessing: Fill NA values with median value of a column
# In[3]:
print("\nMedian of bedrooms column:", df.bedrooms.median())

# Out[3]:
# 4.0

# In[5]:
df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
print("\nData after filling missing values:")
print(df)

# Out[5]:
#    area  bedrooms  age   price
# 0  2600       3.0   20  550000
# 1  3000       4.0   15  565000
# 2  3200       4.0   18  610000
# 3  3600       3.0   30  595000
# 4  4000       5.0    8  760000
# 5  4100       6.0    8  810000

# In[6]:
reg = linear_model.LinearRegression()
reg.fit(df.drop('price', axis='columns'), df.price)

# Out[6]:
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

# In[7]:
print("\nCoefficients:", reg.coef_)

# Out[7]:
# array([ 112.06244194, 23388.88007794, -3231.71790863])

# In[8]:
print("Intercept:", reg.intercept_)

# Out[8]:
# 221323.00186540408

# Find price of home with 3000 sqr ft area, 3 bedrooms, 40 year old
# In[9]:
print("\nPrice for 3000 sqft, 3 bedrooms, 40 years old:", reg.predict([[3000, 3, 40]]))

# Out[9]:
# array([498408.25158031])

# In[10]:
manual_calc = 112.06244194*3000 + 23388.88007794*3 + -3231.71790863*40 + 221323.00186540384
print("Manual calculation verification:", manual_calc)

# Out[10]:
# 498408.25157402386

# Find price of home with 2500 sqr ft area, 4 bedrooms, 5 year old
# In[11]:
print("\nPrice for 2500 sqft, 4 bedrooms, 5 years old:", reg.predict([[2500, 4, 5]]))

# Out[11]:
# array([578876.03748933])

print("\n" + "="*60)
print("Result: Thus, the python program for regression model was executed successfully.")
print("="*60)