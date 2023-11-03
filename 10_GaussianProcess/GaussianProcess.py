import numpy as np
import matplotlib.pyplot as plt

# Implementation of Gaussian Process Regression
class GaussianProcess:
    def __init__(self, kernel, noise):
        self.kernel = kernel
        self.noise = noise
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
        # Calculate covariance matrix
        self.K = self.kernel(X, X)
        self.K += self.noise * np.eye(X.shape[0])
        # Calculate mean and covariance of posterior distribution
        self.K_inv = np.linalg.inv(self.K)
        self.alpha = np.dot(self.K_inv, self.y)
        
    def predict(self, X_star):
        # Calculate mean and variance of predictive distribution
        K_star = self.kernel(self.X, X_star)
        y_pred = np.dot(K_star.T, self.alpha)
        K_star_star = self.kernel(X_star, X_star)
        y_var = K_star_star - np.dot(K_star.T, np.dot(self.K_inv, K_star))
        
        return y_pred.flatten(), np.diag(y_var)
        

# Define the kernel function
def rbf_kernel(X1, X2, length_scale=1.0, scale=1.0):
    dist_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
    return scale * np.exp(-0.5 * dist_sq / length_scale**2)


# Generate synthetic data
np.random.seed(0)
X_train = np.linspace(-5, 5, 10).reshape(-1, 1)
y_train = np.sin(X_train) + np.random.randn(*X_train.shape) * 0.1

# Generate test inputs
X_test = np.linspace(-7, 7, 100).reshape(-1, 1)

# # Data from forex stocks
# from data.forexPandasReader import forexHis
# import random
# dataNum = 150 # number of datas
# dataTestPercent = .7 # test data percentage
# X_train,y_train = forexHis(dataNum)
# X_test , yTest = [] , []
# for i in range(int(dataTestPercent * dataNum)):
#     popIndex = random.randint(0, len(X_train)-1)
#     X_test.append(X_train.pop(popIndex))
#     yTest.append(y_train.pop(popIndex))

# X_train, y_train, X_test = np.array(X_train).reshape(-1,1), np.array(y_train), np.array(X_test).reshape(-1, 1)*100

# Create a Gaussian Process object
gp = GaussianProcess(kernel=rbf_kernel, noise=.1)

# Fit the GP model to the data
gp.fit(X_train, y_train)

# Make predictions
y_pred, y_var = gp.predict(X_test)

# Plot the results
# plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c='r', label='Training Data')
plt.plot(X_test, y_pred, c='b', label='Mean Prediction')
plt.fill_between(X_test.flatten(), y_pred - 2 * np.sqrt(y_var), y_pred + 2 * np.sqrt(y_var),
                 color='gray', alpha=0.3, label='sigma2 Standard Deviations')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gaussian Process Regression')
plt.legend()
plt.grid(True)
plt.show()