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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int makeGradientCov(double[:,:] x, double[:] params, double[:,:] output, double(*kernelDeriv)(double), double[:] xPrime) except -1:
    '''Constructs a covariance matrix for deriving the gradient of the GP at the given
        input locations, writing it to output.

    args:
        x: The matrix of input locations; should be [nSamples x nDimensions].
        params: The parameters of the kernel to use.
        output: An array to write the results to; should be [nDimensions x nDimensions].
        kernelDeriv: A pointer to a function that takes the distance^2 / length^2 and returns the
            the derivative of the covariance with respect to the scaled distance.
            E.g., this is -.5*exp(-.5*scaledDist) for the squared exponential kernel.
        xPrime: The location to compute the gradient.  Should be [nDimensions].

    Returns:
        0 if the calculation successfully wrote to 'output', -1 otherwise.
    '''
    cdef Size n = output.shape[0]
    cdef Size p = x.shape[1]
    cdef Size i, j, k
    cdef double distance
    for i in range(n):
        distance = 0.
        for j in range(p):
            distance += (x[i,j]-xPrime[j])**2
        for j in range(p):
            output[i,j] = 2.*(xPrime[j]-x[i,j])*params[1]*kernelDeriv(distance/(params[0]*params[0]))/(params[0]*params[0])
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

    def __init__(self,x,y,kernel="matern32"):
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
        self.nData = self.x.shape[0]
        self.nDim = self.x.shape[1]
        self.nParams = 3
        self.ready = False
        # Initialize arrays
        self.covChol = np.zeros((self.nData,self.nData))
        self.alpha = np.zeros(self.nData)
        self.params = np.ones(self.nParams)
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
    
    cpdef int precompute(self, double[:] params=None) except -1:
        '''Conducts essential precomputation for the Gaussian Process.
            Specifically, this constructs the covariance matrix, the Cholesky thereof (L), and
            alpha = L.T * L * y.  For certain inputs, the numerics are not very stable, and so 
            the Cholesky operation may incorrectly determine that the covariance matrix
            is not positive definite.  This is best resolved by using a small amount of
            white noise (a diagonal addition to the covariance matrix); see parameter information.
        
        args:
            params (optional): the parameters of the kernel to use.  Otherwise use whatever is
            currently in self.params.
        
        Returns:
            0 if successful, otherwise -1.'''
        if params is not None:
            self.params = params.copy()
        # Construct the (lower diagonal portion of the) covariance matrix
        makeCov(self.x,self.params,self.covChol,self.kernelPtr)
        # Compute the cholesky of the covariance matrix
        choleskyInplace(self.covChol)
        # Compute alpha = L.T^-1 * L^-1 * y
        cdef Size i
        cdef int increment = 1
        cdef int n = self.nData
        for i in range(self.nData):
            self.alpha[i] = self.y[i]
        blas.dtrsv('U','T','N',&n,&self.covChol[0,0],&n,&self.alpha[0],&increment)
        blas.dtrsv('U','N','N',&n,&self.covChol[0,0],&n,&self.alpha[0],&increment)
        self.ready = True
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
        if params is not None or not self.ready:
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
        self.ready = False
        return results
    
    cpdef predict(self, xTest):
        '''Compute the Gaussian process' prediction for a given set of points.
        
        args:
            xTest: the points to make predictions at.  Should be [nPoints x nDimensions].

        Returns:
            A vector of predicted means and the covariance matrix of the predictions.  For
                reference, the marginal uncertainties are np.sqrt(np.diag(var)).
        '''
        # Sanitize inputs
        xTest = np.atleast_2d(np.transpose(xTest)).T.astype(np.double)
        assert xTest.ndim == 2, "xTest must be 1 or 2 dimensional"
        assert self.nDim == xTest.shape[1], "xTest must be [n x p], with p = " + str(self.nDim)
        if not self.ready:
            self.precompute()
        # Compute the test covariance
        KTest = np.empty((self.nData,xTest.shape[0]))
        makeCov(self.x,self.params,KTest,self.kernelPtr,xTest)
        predMean = np.matmul(KTest.T,self.alpha)
        v = solve_triangular(self.covChol,KTest,lower=True)
        predVariance = np.zeros((xTest.shape[0],xTest.shape[0]))
        makeCov(xTest,self.params,predVariance,self.kernelPtr)
        predVariance += np.eye(len(xTest))*self.params[2] - np.matmul(v.T,v)
        return predMean, predVariance


    cpdef gradient(self, xTest):
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
        if not self.ready:
            self.precompute()
        # Compute the test covariance
        KTest = np.empty((self.nData,self.nDim))
        makeGradientCov(self.x,self.params,KTest,self.kernelDerivPtr,xTest)
        return np.matmul(KTest.T,self.alpha)
