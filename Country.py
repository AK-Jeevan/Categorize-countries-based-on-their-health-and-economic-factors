# To categorise the countries using socio-economic and health factors that determine the overall development of the country using K-Means Clustering
# Evaluating the optimal number of clusters using Elbow Method

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

data = pd.read_csv('Country-data.csv')
print(data.head())
print(data.describe())
data.dropna(inplace=True)

# Drop non-numeric columns for clustering
x = data.drop(['country'], axis=1)

# Feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Find optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Let's choose k=3 (for example, based on the elbow plot)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(x_scaled)

# Add cluster labels to the original data
data['Cluster'] = clusters

# Visualize clusters using the first two principal components
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=clusters, palette='Set1', s=100)
plt.title('Country Clusters (PCA-reduced)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.show()

# Show a few countries from each cluster
for i in range(optimal_k):
    print(f"\nCluster {i} countries:")
    print(data[data['Cluster'] == i]['country'].values)