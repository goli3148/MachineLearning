import numpy as np
from sklearn.neighbors import NearestNeighbors

def lle(X, n_components, k, reg=1e-3):
    # Step 1: Find K-nearest neighbors for each data point
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, indices = nbrs.kneighbors(X)
    neighbors = indices[:, 1:]

    # Step 2: Compute the reconstruction weights
    W = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        Xi = X[i] - X[neighbors[i]]
        C = np.dot(Xi, Xi.T) + reg*np.eye(k)
        w = np.linalg.solve(C, np.ones(k))
        w /= np.sum(w)
        W[i, neighbors[i]] = w

    # Step 3: Compute the embedding coordinates
    M = np.eye(X.shape[0]) - W
    eigvals, eigvecs = np.linalg.eig(np.dot(M.T, M))
    indices = np.argsort(eigvals)[1:n_components+1]
    Y = eigvecs[:, indices]
    return Y

# Example usage
X = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [2, 0], [2, 1]])  # Input data points
n_components = 2  # Desired number of dimensions in the lower-dimensional space
k = 3  # Number of nearest neighbors

Y = lle(X, n_components, k)
print(Y)