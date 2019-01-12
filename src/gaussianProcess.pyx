cimport cython

# Kernel functions and their derivatives w.r.t. scaled radius^2
cdef double sqExpKernel(double scaledDist):
    return exp(-.5*scaledDist)

cdef double sqExpKernelDeriv(double scaledDist):
    return -.5*exp(-.5*scaledDist)

cdef double expKernel(double scaledDist):
    return exp(-sqrt(scaledDist))

cdef double matern32Kernel(double scaledDist):
    return (1.+sqrt(3*scaledDist))*exp(-sqrt(3.*scaledDist))

cdef double matern32KernelDeriv(double scaledDist):
    return -1.5*exp(-sqrt(3.*scaledDist))

@cython.cdivision(True)
cdef double matern52Kernel(double scaledDist):
    return (1.+sqrt(5.*scaledDist)+5.*scaledDist/3.)* exp(-sqrt(5*scaledDist))

@cython.cdivision(True)
cdef double matern52KernelDeriv(double scaledDist):
    return -(5./6.)*(1.+sqrt(5*scaledDist))* exp(-sqrt(5*scaledDist))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int makeCov(double[:,:] x, double[:] params, double[:,:] output, double(*kernel)(double) , double[:,:] xPrime=None) except -1:
    '''Constructs a covariance matrix for the given inputs and writes it to output.
        For efficiency reasons, only writes the lower triangular portion of the matrix if
        xPrime is not None.

    args:
        x: The matrix of input locations; should be [nSamples x nDimensions].
        params: The parameters of the kernel to use.
        output: An array to write the results to; should be [nDimensions x nDimensions].
        kernel: A pointer to a function that takes the distance^2 / length^2 and returns the
            a covariance.  E.g., this is exp(-.5*scaledDist) for the squared exponential kernel.
        xPrime: If the goal is to make predictions, this is the locations to predict for.  Should
            be [nPredictions x nDimensions].

    Returns:
        0 if the calculation successfully wrote to 'output', -1 otherwise.
    '''
    cdef Size n = output.shape[0]
    cdef Size m = output.shape[1]
    cdef Size p = x.shape[1]
    cdef Size i, j, k
    cdef Size jMax = m
    cdef bint isSymmetric = False
    cdef double distance

    # Check that the inputs are valid
    if (x is None) or (params is None) or (output is None):
        raise ValueError("Only xPrime may be None.")
    if params.shape[0] != 3:
        raise ValueError("Params must be length 3.")
    if x.shape[0] != output.shape[0]:
        raise ValueError("Output and x have mismatched shapes.")
    if xPrime is None:
        isSymmetric =True
        xPrime = x
    elif (output.shape[1] != xPrime.shape[0]):
        raise ValueError("Output and xPrime have mismatched shapes.")
    elif (xPrime.shape[1] != x.shape[1]):
        raise ValueError("The dimension of x and xPrime differ.")

    # Construct the kernel.
    for i in range(n):
        # If the kernel is symmetric, only bother making the lower triangular part.
        if isSymmetric:
            jMax = i+1
        for j in range(jMax):
            distance = 0
            for k in range(p):
                distance += (x[i,k]-xPrime[j,k])**2
            output[i,j] = params[1]*kernel(distance/(params[0]*params[0]))
            if isSymmetric and (i==j):
                output[i,j] += params[2]
    return 0

# Gaussian Process object
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef class GaussianProcess:
    '''A class for doing Gaussian process computation and modeling.

    Currently supports the following kernels:
        * Exponential
        * Squared Exponential
        * Matern(3/2)
        * Matern(5/2)

    Example:
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
    '''

    def __init__(self,x,y,kernel="squaredExp"):
        '''Initializes the Gaussian process (GP) object with the observations and a kernel type.
        
        args:
            x: The locations of the observations to condition the GP on;
                should be [nSamples x nDimensions].
            y: The values of the input at the locations given by x. Should be [nSamples].
            kernel: The name of the kernel to use.  See help(GaussianProcess) for more
                information.
        
        Returns:
            The Gaussian processes object for further use.'''
        # Sanitize inputs
        self.x = np.atleast_2d(np.transpose(x)).T.astype(np.double)
        self.y = np.atleast_1d(y).astype(np.double)
        assert self.x.shape[0] == self.y.shape[0], \
                "x and y must have the same length first dimension."
        # Record basic information
        self.kernelName = kernel
        self.nData = self.x.shape[0]
        self.nDim = self.x.shape[1]
        self.nParams = 3
        self.choleskyFresh = False
        self.alphaFresh = False
        # Initialize arrays
        self.covChol = np.zeros((self.nData,self.nData))
        self.alpha = np.zeros(self.nData)
        self.params = np.array([1.,np.mean(np.asarray(self.y)**2),1e-4])
        # Match the kernel string to a covariance function
        if kernel.lower() == "squaredexp":
            self.kernelPtr = &sqExpKernel
            self.kernelDerivPtr = &sqExpKernelDeriv
        elif kernel.lower() == "exp":
            self.kernelPtr = &expKernel
            self.kernelDerivPtr = NULL
        elif kernel.lower() == "matern32":
            self.kernelPtr = &matern32Kernel
            self.kernelDerivPtr = &matern32KernelDeriv
        elif kernel.lower() == "matern52":
            self.kernelPtr = &matern52Kernel
            self.kernelDerivPtr = &matern52KernelDeriv
        else: raise ValueError("Kernel name not recognized: "+str(kernel))
        return
    
    cpdef int precompute(self, double[:] params=None, bint force=False) except -1:
        '''Conducts essential precomputation for the Gaussian Process.
            Specifically, this constructs the covariance matrix, the Cholesky thereof (L), and
            alpha = L.T * L * y.  For certain inputs, the numerics are not very stable, and so 
            the Cholesky operation may incorrectly determine that the covariance matrix
            is not positive definite.  This is best resolved by using a small amount of
            white noise (a diagonal addition to the covariance matrix); see parameter information.
        
        args:
            params (optional): the parameters of the kernel to use.  Otherwise use whatever is
                currently in self.params.
            force (default False): Recompute even if cholesky and alpha are flagged as fresh.
                This shouldn't be necessary under normal conditions, since it is automatically
                set to True if params is not None.
        
        Returns:
            0 if successful, otherwise -1.'''
        if params is not None:
            self.params = params.copy()
            force = True
        if not self.choleskyFresh or force:
            # Construct the (lower diagonal portion of the) covariance matrix
            makeCov(self.x,self.params,self.covChol,self.kernelPtr)
            # Compute the cholesky of the covariance matrix
            choleskyInplace(self.covChol)
            self.choleskyFresh = True
            self.alphaFresh = False
        cdef Size i
        cdef int increment = 1
        cdef int n = self.nData
        if not self.alphaFresh or force:
            # Compute alpha = L.T^-1 * L^-1 * y
            for i in range(self.nData):
                self.alpha[i] = self.y[i]
            blas.dtrsv('U','T','N',&n,&self.covChol[0,0],&n,&self.alpha[0],&increment)
            blas.dtrsv('U','N','N',&n,&self.covChol[0,0],&n,&self.alpha[0],&increment)
            self.alphaFresh = True
        return 0

    cpdef double logLikelihood(self, double[:] params=None) except? 999.:
        '''Computes the log-likelihood of the Gaussian process for the given parameters, x, and y.
        
        args:
            params (optional): the parameters of the kernel to use.  Otherwise use whatever is
            currently in self.params.

        Returns:
            The log-likelihood as a double.
        '''
        cdef Size i;
        self.precompute(params)
        cdef double result = -.5*self.nData*log(2.*pi)
        for i in range(self.nData):
            result -= log(self.covChol[i,i]) + .5*self.alpha[i]*self.y[i]
        return result
    
    def optimizeParams(self, paramsGuess=None, tol=1e-3, logBounds=[(-10,10),(-5,5),(-10,10)]):
        '''Attempts to locate the maximum likelihood parameters for the given x and y.
        
        args:
            params (optional): the kernel parameters  to start from.  Otherwise use whatever is
            currently in self.params.
            tol (optional): the error allowed by the optimizer before it is considered to be
                converged.
            logBounds (optional): the bounds on the log of the parameters to be optimized.  This
                is important for avoiding implausible parts of parameter space which would cause
                positive-definite errors in the Cholesky computation.

        Returns:
            The optimized parameters (which are also written to self.params).
        '''
        if paramsGuess is None:
            paramsGuess = self.params
        assert len(paramsGuess) == self.nParams, "paramsGuess must be length " + str(self.nParams)
        results = minimize(lambda p: -self.logLikelihood(np.exp(p)),np.log10(paramsGuess),tol=tol,bounds=logBounds)
        self.params = np.exp(results.x)
        self.choleskyFresh = False
        self.alphaFresh = False
        return results
    
    cpdef predict(self, object xTest, bint getCovariance=True):
        '''Compute the Gaussian process' prediction for a given set of points.
        
        args:
            xTest: the points to make predictions at.  Should be [nPoints x nDimensions].
            getCovariance: Whether to get the predictive covariance as well as the mean.

        Returns:
            A vector of predicted means and (if getCovariance==True),
            the covariance matrix of the predictions.  For reference, the marginal
            uncertainties are np.sqrt(np.diag(var)).
        '''
        # Sanitize inputs
        xTest = np.atleast_2d(np.transpose(xTest)).T.astype(np.double)
        assert xTest.ndim == 2, "xTest must be 1 or 2 dimensional"
        assert self.nDim == xTest.shape[1], "xTest must be [n x p], with p = " + str(self.nDim)
        self.precompute()
        # Compute the test covariance
        KTest = np.empty((self.nData,xTest.shape[0]))
        makeCov(self.x,self.params,KTest,self.kernelPtr,xTest)
        predMean = np.matmul(KTest.T,self.alpha)
        if not getCovariance:
            return predMean
        v = solve_triangular(self.covChol,KTest,lower=True)
        predVariance = np.zeros((xTest.shape[0],xTest.shape[0]))
        makeCov(xTest,self.params,predVariance,self.kernelPtr)
        predVariance += np.eye(len(xTest))*self.params[2] - np.matmul(v.T,v)
        return predMean, predVariance


    cpdef object draw(self, object xTest, Size nDraws=1):
        '''Draws a random vector from the Gaussian process at the specified test points.
        
        args:
            xTest: the points to make predictions at.  Should be [nPoints x nDimensions].
            nDraws: the number of draws to produce.  The first draw is much more
                computationally than subsequent draws.

        Returns:
            A matrix of values sampled from the Gaussian process.  [nDraws x nPoints]
        '''
        cdef Size i
        prediction = self.predict(xTest)
        cdef double[:] predMean = prediction[0]
        cdef double[:,:] predVar = prediction[1]
        output = np.zeros((nDraws,predMean.shape[0]))
        cdef double[:,:] out = output
        choleskyInplace(predVar)
        for i in range(nDraws):
            mvNormalRand(predMean,predVar,out[i],True)
        return output


    cpdef object gradient(self, xTest):
        '''Computes the expected gradient of the Gaussian process at a given point.
            Each component i is dy/dx_i.

        args:
            xTest: the point to measure the gradient at.  Should be [nDimensions].

        Returns
            The components of the gradient as a vector of length [nDimensions].
        '''
        xTest = np.atleast_1d(xTest).astype(np.double)
        if self.kernelDerivPtr == NULL:
            raise ValueError("Selected kernel is not differentiable everywhere.")
        assert xTest.ndim == 1,"xTest must be a 1-dimensional array."
        assert xTest.shape[0] == self.nDim, "xTest must be length " + str(self.nDim)
        output = np.zeros(self.nDim)
        self._gradient_(xTest,output)
        return output


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef int _gradient_(self, double[:] xTest, double[:] output) except -1:
        '''Internal function for expected gradient of the Gaussian process at a given point.

        This function is the non-python internal version of gradient(), which does no
            checking of inputs whatsoever.  When in doubt, use gradient() instead.

        args:
            xTest: the point to measure the gradient at.  Should be [nDimensions].
            output: an array to write the results to.  Should be [nDimensions].

        Returns
            0 if successful, -1 if an exception was raised.
        '''
        cdef Size i, j, k
        cdef double distance, covar
        for j in range(self.nDim):
            output[j] = 0.

        # Make sure the precomputed data is ready.
        self.precompute()
        # Compute the kernel matrix * alpha, write to output
        for i in range(self.nData):
            # Compute the scaled distance
            distance = 0.
            for j in range(self.nDim):
                distance += (self.x[i,j]-xTest[j])**2
            distance /= self.params[0]*self.params[0]
            # Combination kernel computation and matrix-vector multiplication K*alpha
            for j in range(self.nDim):
                # Compute the covariance between data and gradient at xTest
                covar = 2.*(xTest[j]-self.x[i,j])*self.params[1] / (self.params[0]*self.params[0])
                covar *= self.kernelDerivPtr(distance)
                # Gradient is above covariance * alpha
                output[j] += covar * self.alpha[i]
        return 0

    cpdef object setY(self, object newY):
        '''Changes the measured y values in a way that minimizes the required recomputation.
            Specifically, this will need to recompute alpha O(n^2) but not the Cholesky O(n^3)

        args:
            newY: the new values of y to use.  Should be [nPoints].

        Returns:
            None
        '''
        newY = np.atleast_1d(newY).astype(np.double).copy()
        assert self.x.shape[0] == newY.shape[0], \
                "x and y must have the same length first dimension."
        self.y = newY
        self.alphaFresh = False
        return None
