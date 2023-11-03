from typing import Any, Callable, List
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike, NDArray


# Gussian Kernel
class GaussianKernel():
    def gauss_const(self, h):
        return 1/(h*np.sqrt(np.pi*2))
    def gauss_exp(self, ker_x, xi, h):
        num =  - 0.5*np.square((xi- ker_x))
        den = h*h
        return num/den
    def kernel_function(self, h, ker_x, xi)->Any:
        const = self.gauss_const(h)
        gauss_val = const * np.exp(self.gauss_exp(ker_x, xi, h))
        return gauss_val
# Rational Quadratic Kernel
class RationalQuadraticKernel():
    def kernel_function(self, h, ker_x, xi, alpha=1)->Any:
        diff = (xi - ker_x)**2
        form = diff/(2*alpha*(h**2))
        form += 1
        return form ** -alpha
# Squared Exponential Kernel 
class SquaredExponentialKernel():
    def kernel_function(self, h, ker_x, xi)->Any:
        diff = (xi - ker_x)**2
        diff *= -1
        diff /= (2*(h**2))
        return np.exp(diff)
    
# Kernel Regression
# Based on NadaryaWatson
class KernelRegression():
    # dim (example: if equals to one == only one feature)
    dim : int = 1
    
    # Training Data
    xTrainData : List
    yTrainData : List
    # Test Data
    xTest: List
    yTest: List
    
    # Exceptation
    Expectation : List
    # Test function Expectation
    testExpectation: List
    
    # Kernel Class
    kernel: Callable
    H: int = 1
    
    
    def __init__(self, **kwargs) -> None:
        # Get Kwargs
        self.xTrainData = kwargs['xTrain']
        self.yTrainData = kwargs['yTrain']
        
        self.xTest = kwargs['xTest'] if 'xTest' in kwargs else []
        self.yTest = kwargs['yTest'] if 'yTest' in kwargs else []
        
        self.dim = kwargs['dim'] if 'dim' in kwargs else 1
        self.H = kwargs['H'] if 'H' in kwargs else 1
        
        self.Expectation = [0 for _ in range(len(self.xTrainData))]
        self.kernel = kwargs['kernel'] if 'kernel' in kwargs else GaussianKernel().kernel_function
    
    def fit(self):
        # Fit data Using NadaryaWatson
        E = []
        # sigma Operator
        for x in self.xTrainData:
            result1 = 0
            for index in range(len(self.xTrainData)):
                result1 += self.weight(x, self.xTrainData[index])*self.yTrainData[index]
            E.append(result1)
        self.Expectation = E
        # erase the expectation
        E = []
        # calculate expectation for test data
        # sigma operator
        for x in self.xTest:
            result1 = 0
            for index in range(len(self.xTrainData)):
                result1 += self.weight(x, self.xTrainData[index])*self.yTrainData[index]
            E.append(result1)
        self.testExpectation = E
    
    # Calculating Weights
    def weight(self, x, xi):
        numerator = self.kernel(self.H, x, xi)
        denominator = 0
        # sigma Operator
        for xj in self.xTrainData:
            denominator += self.kernel(self.H, x, xj)
        return numerator/denominator
        
    def plotting(self):
        # Plot the results
        plt.figure()
        plt.scatter(self.xTrainData, self.yTrainData, marker = '^', color='b', label = 'train data', zorder=10)
        plt.plot(self.xTrainData, self.Expectation, color='b', label = 'kernel regression', zorder=1)
        plt.scatter(self.xTest, self.testExpectation, marker = '*', color='r', label = 'test prediction',zorder=20)
        plt.scatter(self.xTest, self.yTest, color='g', label = 'test', zorder=30)
        plt.legend()
        plt.show()

        
    
from data import dataGenerator

# EXAMPLE ONE
# x,y = dataGenerator.oneDimEx1()
# kr = KernelRegression(xTrain=x, yTrain=y, H=3)
# kr.fit()
# kr.plotting()

# # EXAMPLE TWO
# x,y = dataGenerator.oneDimEx2()
# kr = KernelRegression(xTrain=x, yTrain=y, H=1)
# kr.fit()
# kr.plotting()

# # EXAMPLE THREE
# x,y = dataGenerator.oneDimEx3()
# kr = KernelRegression(xTrain=x, yTrain=y, H=10)
# kr.fit()
# kr.plotting()

# # EXAMPLE FOUR
# x,y = dataGenerator.forexCurrencies()
# xTest = [x.pop()]
# yTest = [y.pop()]
# kr = KernelRegression(xTrain=x, yTrain=y, H=2, yTest=yTest, xTest=xTest, kernel='SquaredExponentialKernel')
# kr.fit()
# kr.plotting()

# # EXAMPLE FIVE
# x,y = dataGenerator.forexPolyGon()
# xTest = [x.pop()]
# yTest = [y.pop()]
# kr = KernelRegression(xTrain=x, yTrain=y, H=2, yTest=yTest, xTest=xTest, kernel='SquaredExponentialKernel')
# kr.fit()
# kr.plotting()

# EXAMPLE SIX
# getting sample from forex stocks
from data.forexPandasReader import forexHis
import random
dataNum = 100 # number of data for train and test
dataTestPercent = .3 # percent of data for testing
x,y = forexHis(dataNum)
xTest , yTest = [] , []
# split test and train data
for i in range(int(dataTestPercent * dataNum)):
    popIndex = random.randint(0, len(x)-1)
    xTest.append(x.pop(popIndex))
    yTest.append(y.pop(popIndex))
# Kernel Regression with gaussian kernel
kr = KernelRegression(xTrain=x, yTrain=y, H=2, yTest=yTest, xTest=xTest)
kr.fit()
kr.plotting()
# Kernel Regression with Rational Quadratic kernel
kr = KernelRegression(xTrain=x, yTrain=y, H=2, yTest=yTest, xTest=xTest, kernel = RationalQuadraticKernel().kernel_function)
kr.fit()
kr.plotting()
# Kernel Regression with Squared Exponential kernel
kr = KernelRegression(xTrain=x, yTrain=y, H=2, yTest=yTest, xTest=xTest, kernel = SquaredExponentialKernel().kernel_function)
kr.fit()
kr.plotting()
