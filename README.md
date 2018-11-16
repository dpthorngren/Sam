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
* [Scipy](https://www.scipy.org/)
* [Boost.Random](https://www.boost.org/doc/libs/1_68_0/doc/html/boost_random.html)
* [Boost.Special](https://www.boost.org/doc/libs/1_68_0/libs/math/doc/html/special.html)

Once these are installed, Sam is compiled with the following command:

`python setup.py build_ext --inplace`

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
