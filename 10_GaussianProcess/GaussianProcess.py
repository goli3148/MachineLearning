import numpy as np
class Kernel():
    params = []
    def __init__(self, params) -> None:
        self.params = params
    def exponential(self):
        theta1, theta2 = self.params[0], self.params[1]
        x, xp = self.params[2], self.params[3]
        return theta1 * np.exp(-.5 * theta2 * np.subtract.outer(x, xp)**2)
    def gauss(self):
        ...
    def kernel(self, kernelType):
        if kernelType == 'gauss':
            ...
        elif kernelType == 'exponential':
            return self.exponential()


class GaussianProcess():
    def __init__(self) -> None:
        pass