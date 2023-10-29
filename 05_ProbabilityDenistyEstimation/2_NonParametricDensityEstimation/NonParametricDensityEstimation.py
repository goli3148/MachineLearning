from matplotlib import pyplot
from numpy.random import normal
from numpy import hstack, asarray, exp
from sklearn.neighbors import KernelDensity

# generate samples
def generate_samples():
    samples = hstack((normal(loc=20, scale=5, size=300), normal(loc=40, scale=5, size=700)))
    # plot histogram
    
    return samples

samples = generate_samples()
# fit density
model = KernelDensity(bandwidth=2, kernel='gaussian')
samples = samples.reshape((len(samples), 1))
model.fit(samples)
# sample probabilities for a range of outcomes
values = asarray([value for value in range(1, 60)])
values = values.reshape((len(values), 1))
probabilities = model.score_samples(values)
probabilities = exp(probabilities)
# plot the histogram and pdf
pyplot.hist(samples, bins=50, density=True)
pyplot.plot(values[:], probabilities)
pyplot.show()
