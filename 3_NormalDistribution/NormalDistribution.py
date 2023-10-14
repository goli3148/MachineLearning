# sample a normal distribution
def sampling_normal_dist():
    from numpy.random import normal
    # define the distribution
    mu = 50
    sigma = 5
    n = 10
    # generate the sample
    sample = normal(mu, sigma, n)
    print(sample)

# pdf and cdf for a normal distribution
def pdf_cdf_plot_samples():
    from scipy.stats import norm
    from matplotlib import pyplot
    # define distribution parameters
    mu = 50
    sigma = 5
    # create distribution
    dist = norm(mu, sigma)
    # plot pdf
    values = [value for value in range(0, 100)]
    probabilities = [dist.pdf(value) for value in values]
    pyplot.plot(values, probabilities)
    pyplot.show()
    # plot cdf
    cprobs = [dist.cdf(value) for value in values]
    pyplot.plot(values, cprobs)
    pyplot.show()

# calculate the values that define the middle 95%
def middle_95_percentage():
    from scipy.stats import norm
    # define distribution parameters
    mu = 50
    sigma = 5
    # create distribution
    dist = norm(mu, sigma)
    low_end = dist.ppf(0.025)
    high_end = dist.ppf(0.975)
    print('Middle 95%% between %.1f and %.1f' % (low_end, high_end))

sampling_normal_dist()
pdf_cdf_plot_samples()
middle_95_percentage()