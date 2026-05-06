from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=15, method='standard')
X_lle = lle.fit_transform(X_sample)  # X_sample = scaled freMTPL2 features

plt.scatter(X_lle[:,0], X_lle[:,1], c=risk_labels, cmap='RdYlGn', alpha=0.5)
plt.title('LLE: 2D Manifold of Policy Feature Space')
