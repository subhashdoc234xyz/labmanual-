# EXERCISE 8 - IMPLEMENT ENSEMBLING TECHNIQUES (K-MEANS)
# EXACT CODE AS PER MANUAL (PAGES 64-65)

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# In [2]:

dataset = load_iris()
# print(dataset)

# In [3]:

X = pd.DataFrame(dataset.data)
X.columns = ['Sepal Length','Sepal Width','Petal Length','Petal Width']
y = pd.DataFrame(dataset.target)
y.columns = ['Targets']
# print(X)

# In [4]:

plt.figure(figsize=(14,7))
colormap = np.array(['red','lime','black'])

# REAL PLOT
plt.subplot(1,3,1)
plt.scatter(X['Petal Length'],X['Petal Width'],c=colormap[y['Targets']],s=40)
plt.title('Real')

# K-MEANS PLOT
plt.subplot(1,5,6,7,2)
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
predY = np.choose(model.labels_,[0,1,2]).astype(np.int64)
plt.scatter(X['Petal Length'],X['Petal Width'],c=colormap[predY],s=40)
plt.title('KMeans')

# GMM PLOT
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa,columns=X.columns)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs)
plt.subplot(1,3,3)
plt.scatter(X['Petal Length'],X['Petal Width'],c=colormap[y_cluster_gmm],s=40)
plt.title('GMM Classification')

# Out[4]:
# Text(0.5, 1.0, 'GMM Classification')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Result: Thus, the python program for the ensemble techniques using k means plotting was executed successfully.")
print("="*60)