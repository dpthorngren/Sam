Sam
===

Sam is a flexible MCMC sampler for Python and Cython (written in the latter).  It was designed with astrophysics use cases in mind, as an alternative choice when the popular [emcee](http://dfm.io/emcee/current/) algorithm is inappropriate.  It allows for the implementation of multiple MCMC algorithms:

1. [Hamiltonian Monte-Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo)
2. [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)
3. [Adaptive Metropolis](https://projecteuclid.org/euclid.bj/1080222083)
4. [Gibbs](https://en.wikipedia.org/wiki/Gibbs_sampling)

The sampler is working and usable, but development is still ongoing.  In particular, a Gaussian process HMC surrogate sampler is planned, and the whole project requires considerably more documentation.

Installation
------------
Sam requires the following libraries to compile:

* [Cython](https://cython.org/)
* [Numpy](http://www.numpy.org/)
* [Boost](https://www.boost.org/) -- specifically the Random, Special, and Math libraries.

Once these are installed, you can install any remaining missing python dependencies (Scipy and Multiprocessing) and compile Sam with the following command from the Sam directory:

`pip install --user .`

If you prefer not to use pip, you may instead use:

`python setup.py install --user`

Finally, for a system-wide install, you may omit the --user, although this may require elevated user privileges.

Example Usage
-------------
Documentation is still a work in progress, but here are a couple of examples.  For now, we consider a very simple case, sampling from a normal distribution:
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
