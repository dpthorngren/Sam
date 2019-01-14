=========
Sam Class
=========

.. module:: sam

.. autoclass:: Sam

Running the Sampler
===================
.. autofunction:: sam.Sam.run

Picking Samplers
================
By default, Sam will use a simple Metropolis-Hastings sampler.  You can manage
the different samplers by using the following functions.  Note that you can
several samplers in combination, so as to use different algorithms for
different parameters.  This is done using the ``dStart`` and ``dStop``

.. autofunction:: sam.Sam.addMetropolis
.. autofunction:: sam.Sam.addAdaptiveMetropolis
.. autofunction:: sam.Sam.addHMC
.. autofunction:: sam.Sam.printSamplers
.. autofunction:: sam.Sam.clearSamplers

Surrogate Sampling
==================
Sam supports the use of a Gaussian process (GP) as a surrogate model for the
posterior probability.  This means that each time the sampler needs to know
the value of the posterior probability at a point, the GP is consulted first.
If the GP is able to estimate the posterior probability with uncertainty less
than a certain tolerance, then this estimate is used and the ``logProbability``
function provided by the user is not called.  Otherwise ``logProbability`` is
called and the result is added to the GP for future reference.  This
approach can reduce the number of likelihood evaluations considerably at the
cost of some modest overhead.  It becomes less effective for high-dimensional
problems.  When the surrogate is enabled, you can access it via
``Sam.surrogate`` to e.g. retrieve the results of actual likelihood
evaluations ( ``Sam.surrogate.x`` and ``Sam.surrogate.y``)

.. autofunction:: sam.Sam.enableSurrogate
.. autofunction:: sam.Sam.disableSurrogate

Examining the Results
=====================
.. autodata:: sam.Sam.samples
    :annotation: = the samples collected as a Numpy ndarray of shape [nThreads, nSamples, nParameters]
.. autodata:: sam.Sam.results
    :annotation: = same as ``samples``, but flattend across threads: [(nThreads x nSamples), nParameters]
.. autofunction:: sam.Sam.summary
.. autofunction:: sam.Sam.getStats

Model Comparison
================
Several functions are available for computing model comparison statistics on
the results of a run.  Since only the posterior probability is known to the
sampler, you must provide a function which returns the prior probability given
the parameters in order to compute these.

.. autofunction:: sam.Sam.getAIC
.. autofunction:: sam.Sam.getBIC
.. autofunction:: sam.Sam.getDIC
