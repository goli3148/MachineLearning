# Example of parametric probabilty density estimation
from numpy.random import normal, randint
from numpy import mean, std
from scipy.stats import norm
from matplotlib import pyplot
# create samples
samples = []
def create_samples():
    global samples
    samples = normal(loc=50, scale=5, size=100)
    # make samples closer to realtiy
    random_int = randint(-2, 2, size=100)
    for index in range(len(samples)):
        samples[index] = samples[index] + random_int[index]

def estimation():
    # calculate parameters
    sample_mean = mean(samples)
    sample_std = std(samples)
    # define distribution
    dist = norm(sample_mean, sample_std)
    # sample probabilities for a range of outcome
    values = [value for value in range(30,70)]
    probabilities = [dist.pdf(value) for value in values]
    # plot the histogram and pdf
    pyplot.hist(samples, bins=10, density=True)
    pyplot.plot(values, probabilities)
    pyplot.show()
    
create_samples()
estimation()