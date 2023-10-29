# sampling exponential distribution
def sampling_exponential_dist():
    from numpy.random import exponential
    # define the distribution
    beta = 50
    n = 10
    # generate the sample
    sample = exponential(beta, n)
    print(sample)

# pdf and cdf for an exponential distribution
def pdf_cdf_plot_samples():
    from scipy.stats import expon
    from matplotlib import pyplot
    #define distribution parameters
    beta = 50
    # create distribution
    dist = expon(beta)
    # plot pdf
    values = [value for value in range(50, 70)]
    probablities = [dist.pdf(value) for value in values]
    pyplot.plot(values, probablities)
    pyplot.show()
    # plot cdf
    cprobs = [dist.cdf(value) for value in values]
    pyplot.plot(values, cprobs)
    pyplot.show()

sampling_exponential_dist()
pdf_cdf_plot_samples()