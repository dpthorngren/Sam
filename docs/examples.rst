========
Examples
========

We start with a barebones example of sampling from a normal distribution with the default Metropolis sampler.

.. code-block:: python

    import sam

    def logProb(x):
        return sam.normalLogPDF(x[0],3.5,1.2)

    s = sam.Sam(logProb,scale=[.5])
    s.run(100000,x0=[0.])
    s.summary()

    # Sampling: <==========> (100000 / 100000)          
    # Dim. Accept    GR* |       Mean       Std. |        16%        50%        84%
    # 0     86.7%  1.001 |       3.47      1.204 |      2.277      3.465      4.667

Here is an example where we fit a logistic regression model to some data using an adaptive Metropolis sampler on four parallel chains:

.. code-block:: python

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
