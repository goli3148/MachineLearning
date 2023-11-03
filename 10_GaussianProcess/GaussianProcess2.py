import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal



def calculate_covariance_matrix(X, kernel_func, theta):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i][j] = kernel_func(X[i], X[j], theta)
    return K



def squared_exponential_kernel(x1, x2, theta):
    return np.exp(-0.5 * np.linalg.norm(x1 - x2)**2 / (theta**2))



X_train = np.array([-2, -1, 0, 2, 4])  # Training inputs
y_train = np.array([4, 1, 0, 1, -2])  # Training outputs



X_test = np.arange(-5, 5, 0.1)  # Test inputs
theta = 0.5  # Kernel parameter

K = calculate_covariance_matrix(X_train, squared_exponential_kernel, theta)
K_star = calculate_covariance_matrix(X_test, squared_exponential_kernel, theta)
K_star_star = calculate_covariance_matrix(X_test, squared_exponential_kernel, theta)
K_inverse = np.linalg.inv(K)

print(K_star.shape)
print(K_inverse.shape)
print(y_train.shape)


mean = K_star @ K_inverse @ y_train
covariance = K_star_star - K_star @ K_inverse @ K_star.T


plt.figure(figsize=(10, 6))
plt.errorbar(X_test, mean, yerr=np.diag(covariance), fmt='b-', label="Predicted")
plt.scatter(X_train, y_train, color='r', label="Training Data")
plt.legend()
plt.show()