from matplotlib import pyplot as plt
import numpy as np
from numpy.random import normal

def calculate_covariance_matrix(X, kernel_func, theta):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i][j] = kernel_func(X[i], X[j], theta)
    return K
    
def squared_exponential_kernel(x1, x2, theta):
    return np.exp(-0.5 * np.linalg.norm(x1 - x2)**2 / (theta**2))


x_train = np.array([-2, -1, 0, 2, 4])
y_train = np.zeros((x_train.shape[0], x_train.shape[0]))

# Sampling from gp
x_star = np.arange(-5, 5, .5)
y_star = np.zeros((x_star.shape[0], x_star.shape[0]))
K = calculate_covariance_matrix(x_star, squared_exponential_kernel, .5)
for i in range(x_star.shape[0]):
    y_star[i]=normal(loc=0., scale=K[i])
plt.plot(x_star, y_star)
plt.show()