# example of simulating a binomial process and counting success
def binomail_dist():
    from numpy.random import binomial
    # define the parameters of the distribution
    p = 0.3
    k = 100
    # run a single simulation
    success = binomial(k, p)
    print('Total Success: %d' % success)

# calculate moments of a binomial distribution
def binomail_dist_2():
    from scipy.stats import binom
    # define the parameters of the distribution
    p = 0.3
    k = 100
    # calculate moments
    mean, var, _, _ = binom.stats(k, p, moments='mvsk')
    print('Mean=%.3f, Variance=%.3f' % (mean, var))

# example of using the pmf for the binomial distribution
def binomail_dist_pmf():
    from scipy.stats import binom
    # define the parameters of the distribution
    p = 0.3
    k = 100
    # define the distribution
    dist = binom(k, p)
    # calculate the probability of n successes
    for n in range(10, 110, 10):
        print('P of %d success: %.3f%%' % (n, dist.pmf(n)*100))

binomail_dist()
binomail_dist_2()
binomail_dist_pmf()