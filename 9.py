# EXERCISE 9 - K-MEANS CLUSTERING ALGORITHM
# EXACT CODE AS PER MANUAL (PAGES 66-73) - CORRECTED FOR PYTHON SCRIPT

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Remove %matplotlib inline - this is only for Jupyter notebooks
# Use plt.show() instead to display plots

print("="*60)
print("EXERCISE 9: K-MEANS CLUSTERING ALGORITHM")
print("="*60)

# Read the CSV file
df = pd.read_csv("income.csv")
print("\nFirst 5 rows of data:")
print(df.head())

# Plot without clustering
plt.figure(figsize=(8, 6))
plt.scatter(df.Age, df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.title('Income vs Age (Before Clustering)')
plt.show()

# Apply K-Means clustering
km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(df[['Age','Income($)']])
print("\nPredictions:", y_predicted)

# Add cluster column to dataframe
df['cluster'] = y_predicted
print("\nData with clusters:")
print(df.head())

# Print cluster centers
print("\nCluster Centers:")
print(km.cluster_centers_)

# Plot clusters
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.figure(figsize=(8, 6))
plt.scatter(df1.Age, df1['Income($)'], color='green', label='Cluster 0')
plt.scatter(df2.Age, df2['Income($)'], color='red', label='Cluster 1')
plt.scatter(df3.Age, df3['Income($)'], color='black', label='Cluster 2')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='purple', marker='*', s=200, label='centroid')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()
plt.title('K-Means Clustering')
plt.show()

print("\n" + "="*60)
print("Preprocessing using min max scaler")
print("="*60)

# Preprocessing using min max scaler
scaler = MinMaxScaler()
df['Income($)'] = scaler.fit_transform(df[['Income($)']])
df['Age'] = scaler.fit_transform(df[['Age']])

print("\nScaled data (first 5 rows):")
print(df.head())

# Plot scaled data
plt.figure(figsize=(8, 6))
plt.scatter(df.Age, df['Income($)'])
plt.xlabel('Age (scaled)')
plt.ylabel('Income($) (scaled)')
plt.title('Scaled Data')
plt.show()

# Apply K-Means on scaled data
km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(df[['Age','Income($)']])
print("\nPredictions on scaled data:", y_predicted)

# Add cluster column
df['cluster'] = y_predicted
print("\nScaled data with clusters:")
print(df.head())

# Print cluster centers for scaled data
print("\nCluster Centers (scaled):")
print(km.cluster_centers_)

# Plot clusters on scaled data
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.figure(figsize=(8, 6))
plt.scatter(df1.Age, df1['Income($)'], color='green', label='Cluster 0')
plt.scatter(df2.Age, df2['Income($)'], color='red', label='Cluster 1')
plt.scatter(df3.Age, df3['Income($)'], color='black', label='Cluster 2')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='purple', marker='*', s=200, label='centroid')
plt.xlabel('Age (scaled)')
plt.ylabel('Income($) (scaled)')
plt.legend()
plt.title('K-Means Clustering on Scaled Data')
plt.show()

print("\n" + "="*60)
print("Elbow Plot")
print("="*60)

# Elbow Plot
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)

plt.figure(figsize=(8, 6))
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse, 'bo-')
plt.title('Elbow Method for Optimal K')
plt.show()

print("\n" + "="*60)
print("Result: Thus, the python program for clustering algorithm using k-means was executed successfully.")
print("="*60)