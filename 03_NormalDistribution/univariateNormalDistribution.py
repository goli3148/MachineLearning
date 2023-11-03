import numpy as np
import matplotlib.pyplot as plt
def univariate_normal(x, mean, variance):
    """pdf of the univariate normal distribution."""
    return ((1. / np.sqrt(2 * np.pi * variance)) * 
            np.exp(-(x - mean)**2 / (2 * variance)))

data = [i for i in range(-10, 10)]
y = []
y2 = []
for i in data:
    y.append(univariate_normal(i, 0, 10))
    y2.append(univariate_normal(i, 0, 20))
plot = plt
plot.plot(data, y)
plot.plot(data, y2)
plot.show()