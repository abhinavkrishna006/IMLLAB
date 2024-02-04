import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
'''
np.random.seed(0)
X = np.random.randn(100, 2)
'''
iris = datasets.load_iris()
X = iris.data

plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.title('Generated Dataset')
plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.show()
