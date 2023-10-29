import numpy as np
import matplotlib.pyplot as plt

class GaussianKernel():
    def gauss_const(self, h):
        return 1/(h*np.sqrt(np.pi*2))
    def gauss_exp(self, ker_x, xi, h):
        num =  - 0.5*np.square((xi- ker_x))
        den = h*h
        return num/den
    def kernel_function(self, h, ker_x, xi):
        const = self.gauss_const(h)
        gauss_val = const * np.exp(self.gauss_exp(ker_x, xi, h))
        return gauss_val
class RationalQuadraticKernel():
    def kernel_function(self, h, ker_x, xi, alpha=1):
        diff = (xi - ker_x)**2
        form = diff/(2*alpha*(h**2))
        form += 1
        return form ** -alpha 
class SquaredExponentialKernel():
    def kernel_function(self, h, ker_x, xi):
        diff = (xi - ker_x)**2
        diff *= -1
        diff /= (2*(h**2))
        return np.exp(diff)
    

class KernelRegression():
    # dim (example: if equals to one == only one feature)
    dim : int = 1
    
    xTrainData : np.array(np.array(float))
    yTrainData : np.array(float)
    xTest: np.array(np.array(float))
    yTest: np.array(float)
    
    Expectation : np.array(float)
    testExpectation: np.array(float)
    
    kernel =  0
    H: int = 1
    
    
    def __init__(self, **kwargs) -> None:
        self.xTrainData = kwargs['xTrain']
        self.yTrainData = kwargs['yTrain']
        
        self.xTest = kwargs['xTest'] if 'xTest' in kwargs else []
        self.yTest = kwargs['yTest'] if 'yTest' in kwargs else []
        
        self.dim = kwargs['dim'] if 'dim' in kwargs else 1
        self.H = kwargs['H'] if 'H' in kwargs else 1
        
        self.Expectation = [0 for _ in range(len(self.xTrainData))]
        if 'kernel' in kwargs:
            if kwargs['kernel'] == "RationalQuadratic":
                self.kernel = RationalQuadraticKernel()
            if kwargs['kernel'] == "SquaredExponentialKernel":
                self.kernel = SquaredExponentialKernel()
        else:
            self.kernel = GaussianKernel()
    
    def weight(self, x, xi):
        numerator = self.kernel.kernel_function(self.H, x, xi)
        denominator = 0
        for xj in self.xTrainData:
            denominator += self.kernel.kernel_function(self.H, x, xj)
        return numerator/denominator
    
    def NadarayaWatson(self):
        E = []
        for x in self.xTrainData:
            result1 = 0
            for index in range(len(self.xTrainData)):
                result1 += self.weight(x, self.xTrainData[index])*self.yTrainData[index]
            E.append(result1)
        self.Expectation = E
        E = []
        for x in self.xTest:
            result1 = 0
            for index in range(len(self.xTrainData)):
                result1 += self.weight(x, self.xTrainData[index])*self.yTrainData[index]
            E.append(result1)
        self.testExpectation = E
        
        
    def plotting(self):
        plt.scatter(self.xTrainData, self.yTrainData, color='b')
        plt.scatter(self.xTest, self.testExpectation, color='r')
        plt.scatter(self.xTest, self.yTest, color='g')
        plt.plot(self.xTrainData, self.Expectation, color='b')
        plt.show()

        
    
from data import dataGenerator

# EXAMPLE ONE
# x,y = dataGenerator.oneDimEx1()
# kr = KernelRegression(xTrain=x, yTrain=y, H=3)
# kr.NadarayaWatson()
# kr.plotting()

# # EXAMPLE TWO
# x,y = dataGenerator.oneDimEx2()
# kr = KernelRegression(xTrain=x, yTrain=y, H=1)
# kr.NadarayaWatson()
# kr.plotting()

# # EXAMPLE THREE
# x,y = dataGenerator.oneDimEx3()
# kr = KernelRegression(xTrain=x, yTrain=y, H=10)
# kr.NadarayaWatson()
# kr.plotting()

# # EXAMPLE FOUR
# x,y = dataGenerator.forexCurrencies()
# xTest = [x.pop()]
# yTest = [y.pop()]
# kr = KernelRegression(xTrain=x, yTrain=y, H=2, yTest=yTest, xTest=xTest, kernel='SquaredExponentialKernel')
# kr.NadarayaWatson()
# kr.plotting()

# # EXAMPLE FIVE
# x,y = dataGenerator.forexPolyGon()
# xTest = [x.pop()]
# yTest = [y.pop()]
# kr = KernelRegression(xTrain=x, yTrain=y, H=2, yTest=yTest, xTest=xTest, kernel='SquaredExponentialKernel')
# kr.NadarayaWatson()
# kr.plotting()

# EXAMPLE SIX
from data.forexPandasReader import forexHis
x,y = forexHis(100)
xTest = [x.pop()]
yTest = [y.pop()]
kr = KernelRegression(xTrain=x, yTrain=y, H=1, yTest=yTest, xTest=xTest)
kr.NadarayaWatson()
kr.plotting()
