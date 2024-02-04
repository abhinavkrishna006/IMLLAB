import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.concatenate([np.random.normal(0, 1, 300), np.random.normal(5, 1, 200)])
data = data.reshape(-1, 1)

plt.hist(data, bins=50, density=True, alpha=0.5, color='b')

gmm = GaussianMixture(n_components=2, random_state=42)
data = data.reshape(-1, 1)
gmm.fit(data)

x = np.linspace(-5, 10, 1000).reshape(-1, 1)
y = np.exp(gmm.score_samples(x))
plt.plot(x, y, '--', label='Gaussian Mixture Model')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()
