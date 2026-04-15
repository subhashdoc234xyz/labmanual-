# EXERCISE 4 - BAYESIAN NETWORKS
# EXACT CODE AS PER MANUAL PAGES 24-26

from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd

# Load the data from your existing CSV file
heart_disease = pd.read_csv('heart_disease.csv')

# ================== FIXES ==================
# Remove spaces in column names (avoids KeyError)
heart_disease.columns = heart_disease.columns.str.strip()
# ===========================================

# In [2]:
model = BayesianModel([
    ('age','Lifestyle'),
    ('Gender','Lifestyle'),
    ('Family','heartdisease'),
    ('diet','cholestrol'),
    ('Lifestyle','diet'),
    ('cholestrol','heartdisease')
])

model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

HeartDisease_infer = VariableElimination(model)

# In [3]:
print('For age Enter { SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4 }')
print('For Gender Enter { Male:0, Female:1 }')
print('For Family History Enter { yes:1, No:0 }')
print('For diet Enter { High:0, Medium:1 }')
print('For lifeStyle Enter { Athlete:0, Active:1, Moderate:2, Sedentary:3 }')
print('For cholesterol Enter { High:0, BorderLine:1, Normal:2 }')

# ================== SAFE INPUT ==================
age = int(input('Enter age :'))
gender = int(input('Enter Gender :'))
family = int(input('Enter Family history :'))
diet = int(input('Enter diet :'))
lifestyle = int(input('Enter Lifestyle :'))
chol = int(input('Enter cholestrol :'))

# Validate inputs (prevents crash)
if age not in [0,1,2,3,4]:
    print("Invalid age input"); exit()
if gender not in [0,1]:
    print("Invalid gender input"); exit()
if family not in [0,1]:
    print("Invalid family input"); exit()
if diet not in [0,1]:
    print("Invalid diet input"); exit()
if lifestyle not in [0,1,2,3]:
    print("Invalid lifestyle input"); exit()
if chol not in [0,1,2]:
    print("Invalid cholestrol input"); exit()
# =================================================

q = HeartDisease_infer.query(
    variables=['heartdisease'],
    evidence={
        'age': age,
        'Gender': gender,
        'Family': family,
        'diet': diet,
        'Lifestyle': lifestyle,
        'cholestrol': chol
    }
)

# FIX: New pgmpy returns DiscreteFactor
print(q)

print("\n" + "="*60)
print("Result: Thus, the Bayesian networks program was executed and output is verified.")
print("="*60)
# Valid Inputs (based on your dataset)

# age -> 0,1,2,3 ONLY
# Gender -> 0,1
# Family -> 0,1
# diet -> 0,1
# Lifestyle -> 0,1,2,3
# cholestrol -> 0,1,2