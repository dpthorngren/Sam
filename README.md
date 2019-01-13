Sam
===

Sam is a flexible MCMC sampler for Python and Cython (written in the latter).  It was designed with astrophysics use cases in mind, as an alternative choice when the popular [emcee](http://dfm.io/emcee/current/) algorithm is inappropriate.  It allows for the implementation of multiple MCMC algorithms:

1. [Hamiltonian Monte-Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo)
2. [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)
3. [Adaptive Metropolis](https://projecteuclid.org/euclid.bj/1080222083)
4. [Gibbs](https://en.wikipedia.org/wiki/Gibbs_sampling)

The sampler is working and usable, but development is still ongoing.  In particular, a Gaussian process HMC surrogate sampler is planned, and the whole project requires considerably more documentation.

Documentation
------------
The full documentation can be found on Readthedocs at [https://sam-mcmc.readthedocs.io](https://sam-mcmc.readthedocs.io).

Installation
------------
Sam requires the following libraries to compile:

* [Cython](https://cython.org/)
* [Numpy](http://www.numpy.org/)
* [Boost](https://www.boost.org/) -- specifically the Random, Special, and Math libraries.
* The Python dev libraries (e.g. python-dev or python3-dev)

Once these are installed, you can install any remaining missing python dependencies (Scipy and Multiprocessing) and compile Sam with the following command from the Sam directory:

`pip install --user .`

If you prefer not to use pip, you may instead use:

`python setup.py install --user`

Finally, for a system-wide install, you may omit the --user, although this may require elevated user privileges.

Basic usage
-------------
1. **Define the log probability** of the target distribution (the posterior).  This is the difficult part, for which a proper discussion is well outside the scope of these instructions.  Consider [Gelman's book](http://www.stat.columbia.edu/~gelman/book/) for a good overview of Bayesian statistics.  Once you know what you want, you should create a function that takes a vector of parameters and returns the log probability.  You will likely want to use either the [Scipy Statistics](https://docs.scipy.org/doc/scipy/reference/stats.html) functions or the builtin Sam logPDF functions.  For example:

```python
import sam
from scipy.stats import norm

def logProb(theta):
    likelihood = norm.logPDF(data,theta[0],theta[1])
    prior = sam.normalLogPDF(theta[0],0,2) + sam.exponentalLogPDF(theta[1],3)
    return likelihood + prior
```

2. **Initialize the sampler**.  You will need to provide the log probability function as well as a scale vector, which is a tuning parameter of the same length as the parameter vector.  This serves different purposes depending on the sampler, but usually you can use a rough guess of the posterior standard deviation.  In Metropolis-Hastings sampler, for example, this sets the default proposal standard deviation.  You can optionally pick a sampler to use at this point, or stick with the default Metropolis sampler.

```python
s = sam.Sam(logProb,[1.,1.])
# Optionally:
# s.addAdaptiveMetropolis()
# s.addMetrpolis(proposalCovariance)
# s.addHMC(steps=10,stepSize=.1)
```

3. **Run the sampler**.  You must provide an initial position to begin sampling from.  You may run several identical samplers in parallel using the ```threads``` keyword.

```python
s.run([1.,1.],samples=10000,burnIn=1000,thinning=10,threads=2)
```

4. **Examine the results**.  Once sampling is complete, the samples can be accessed as ```s.samples```, and ```s.results``` (for the thread-flattened version -- these will be the same if only one thread was used).  Sam provides a few functions to assist in saving the results and checking for convergence.  First, calling ```s.summary()``` prints a brief summary of the posterior sample for each dimension, including the acceptance fraction, Gelman-Rubin diagnostic, mean, standard deviation, and some percentiles.  Additionally ```s.save("filename.npz")``` can be used to save the samples and supporting information to a [Numpy .npz file](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.savez.html).

Examples
-------------
Documentation is still a work in progress, but here are a couple of examples.  First, consider a very simple case, sampling from a normal distribution:
```python
import sam

def logProb(x):
    return sam.normalLogPDF(x[0],3.5,1.2)

s = sam.Sam(logProb,scale=[.5])
s.run(100000,x0=[0.])
s.summary()

# Sampling: <==========> (100000 / 100000)          
# Dim. Accept    GR* |       Mean       Std. |        16%        50%        84%
# 0     86.7%  1.001 |       3.47      1.204 |      2.277      3.465      4.667
```

For a more involved example, here is an example where we fit a [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) model to some data using an adaptive Metropolis sampler on four parallel chains:
```python
import sam
from scipy.special import expit, logit
from scipy.stats import bernoulli

# Create data to use for sampling
betaTrue = np.array([7.5,-1.2])
n = 100
x = column_stack([ones(n),10*np.random.rand(n)])
y = bernoulli.rvs(expit(np.matmul(x,betaTrue)))

def logProb(beta):
    logLikeliood = bernoulli.logpmf(y,expit(np.matmul(x,beta))).sum()
    logPrior = norm.logpdf(beta,[0,0],5.).sum()
    return logLikeliood + logPrior

# Run the MCMC
s = sam.Sam(logProb,[1.,1.])
s.addAdaptiveMetropolis()
s.run(10000,x0=[1.,1.],burnIn=1000,thinning=10,threads=4)
s.summary()

# Sampling: <==========> (10000 / 10000)          
# Sampling: <==========> (10000 / 10000)          
# Sampling: <==========> (10000 / 10000)          
# Sampling: <==========> (10000 / 10000)          
# Dim. Accept     GR |       Mean       Std. |        16%        50%        84%
# 0      5.9%  1.001 |      7.914      1.626 |       6.33      7.761      9.544
# 1      5.9%  1.001 |     -1.144     0.2328 |     -1.374     -1.124    -0.9185
```
