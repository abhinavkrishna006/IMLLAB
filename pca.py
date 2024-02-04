import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
data = iris.data
target_labels = iris.target

pca = PCA(n_components=4)
principal_components = pca.fit_transform(data)


plt.scatter(principal_components[:, 0], principal_components[:, 1], c=target_labels, cmap='viridis')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris dataset after PCA (4 components)")
plt.show()
print("Eigenvalues:", pca.explained_variance_)
