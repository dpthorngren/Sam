======================
Gaussian Process Class
======================

.. module:: sam
.. autoclass:: GaussianProcess
    :members:

Example Use
===========

.. code-block:: python

	import numpy as np
	from matplotlib import pyplot as plt
	from sam import GaussianProcess

	x = np.linspace(0,10,10)
	y = np.sin(x)

	f = GaussianProcess(x,y)
	f.optimizeParams(5*ones(3))

	xTest = np.linspace(0,10,100)
	yTest, yErr = f.predict(xTest)
	yErr = np.sqrt(np.diag(yErr))
	plt.plot(xTest,yTest)
	plt.fill_between(xTest,yTest-yErr,yTest+yErr,alpha=.5)
	plt.plot(x,y,'.')
