from typing import List
import numpy as np
from matplotlib import pyplot
class LinearRegression():
    #
    dim: int
    
    # a vector with length trainSize
    yTrain = np.array(int)
    
    # a matrix with size of dimension * trainsize+1
    xTrain= np.array(np.array(int))
    xTrainFixed = np.array(np.array(int))
    # a vector with length of trainSize
    weight: np.array(int)
    
    # finalFunc
    finalFunc = np.array(np.array)
    
    plot: pyplot
    def __init__(self, xTrain, yTrain, dim=1) -> None:
        self.plot = pyplot
        self.dim = dim
        self.xTrainFixed = xTrain
        self.xTrain = np.append(np.array([1 for _ in range(len(xTrain[0]))]), np.array(xTrain))
        self.xTrain = self.xTrain.reshape(dim+1, len(xTrain[0])).transpose()
        self.yTrain = np.array(yTrain).transpose()
        
    def solve(self) -> np.array(np.array(int)):
        # W = ((X^T X)^-1)(X^T)(Y)
        self.weight = np.matmul(self.xTrain.transpose(), self.xTrain)
        self.weight = np.matmul(np.linalg.inv(self.weight), self.xTrain.transpose())
        self.weight = np.matmul(self.weight, self.yTrain)
        # finalFunc = W^T * X
        self.finalFunc = np.multiply(self.weight.transpose(), self.xTrain)
        return self.finalFunc
    
    def plotting(self):
        yPoints = []
        if self.dim == 1:
            self.xTrainFixed = self.xTrainFixed[0]
            self.plot.scatter(self.xTrainFixed, self.yTrain)
            for i in self.finalFunc:
                yPoints.append(i[0]+i[1])
            self.plot.plot(self.xTrainFixed, yPoints)
            self.plot.show()
        elif self.dim == 2:
            self.plot = pyplot.axes(projection='3d')
            x1 = self.xTrainFixed[0]
            x2 = self.xTrainFixed[1]
            self.plot.scatter3D(x1, x2, self.yTrain)
            for i in self.finalFunc:
                yPoints.append(i[0]+i[1]+i[2])
            self.plot.plot3D(x1, x2, yPoints, color='r')
            pyplot.show()
    

from data import dataGenerator

x,y = dataGenerator.twoDimEx1()
li = LinearRegression(x, y, 2)
li.solve()
li.plotting()

x,y = dataGenerator.oneDimEx1()
x = [x]
li = LinearRegression(x, y, 1)
li.solve()
li.plotting()

x,y = dataGenerator.oneDimEx3()
x = [x]
li = LinearRegression(x, y)
li.solve()
li.plotting()