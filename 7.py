# Ex.No:7 - Build SVM Models
# Exact code from Arjun College of Technology CS3491 Lab Manual (pages 50-63)
# FIXED: Graphs now pop up exactly like in your manual

import matplotlib
matplotlib.use('TkAgg')   # This forces graph windows to appear
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load your csv (must have multiple Label values)
data = pd.read_csv('covid19.csv')

# FIX: Label column has only 1 class — derive labels from infection_order so SVM has 2 classes
data['Label'] = (data['infection_order'] == 1).astype(int)

print("=== First 5 rows of dataset ===")
print(data.head())
print("\n")

print("=== Unique values in each column ===")
for col in data.columns:
    print(col, data[col].unique())
print("\n")

print("=== Value counts of each column ===")
for col in data.columns:
    print(data[col].value_counts())
print("\n")

# First Heatmap (Before Encoding)
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Before Encoding)")
plt.show()

# Label Encoding
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object' or data[col].dtype == 'string':
        data[col] = le.fit_transform(data[col].astype(str))

print("=== First 5 rows AFTER Label Encoding ===")
print(data.head())
print("\n")

# Second Heatmap (After Encoding)
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (After Encoding)")
plt.show()

# Prepare X and y
X = data.drop('Label', axis=1)
y = data['Label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# SVM Model
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Results
print("Accuracy for Runlengthsvm:", metrics.accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC Curve
ns_probs = [0 for _ in range(len(y_test))]
lr_auc = roc_auc_score(y_test, y_pred)
ns_auc = roc_auc_score(y_test, ns_probs)

print("\nROC AUC Score:", lr_auc)
print('No Skill: ROC AUC=%.3f' % ns_auc)
print('SVM: ROC AUC=%.3f' % lr_auc)

# Plot ROC Curve (exactly like your manual screenshot)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)

plt.figure()                          # ← FIXED: forces a fresh figure window
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='SVM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve - SVM Model')
plt.show()

print("\n=== Program executed successfully ===")
