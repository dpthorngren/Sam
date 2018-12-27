# Kernel functions and derivatives
cdef double sqExpKernel(double scaledDist)
cdef double sqExpKernelDeriv(double scaledDist)
cdef double expKernel(double scaledDist)
cdef double matern32Kernel(double scaledDist)
cdef double matern32KernelDeriv(double scaledDist)
cdef double matern52Kernel(double scaledDist)
cdef double matern52KernelDeriv(double scaledDist)

# Covariance function builders
cdef int makeCov(double[:,:] x, double[:] params, double[:,:] output, double(*kernel)(double) , double[:,:] xPrime=?) except -1
cdef int makeGradientCov(double[:,:] x, double[:] params, double[:,:] output, double(*kernelDeriv)(double), double[:] xPrime) except -1

cdef class GaussianProcess:
    cdef readonly double[:,:] x
    cdef readonly double[:] y
    cdef readonly double[:] params
    cdef readonly double[:,:] covChol
    cdef double[:] alpha
    cdef bint ready
    cdef double (*kernelPtr)(double)
    cdef double (*kernelDerivPtr)(double)
    cdef readonly Size nParams
    cdef readonly Size nDim
    cdef readonly Size nData

    cpdef int precompute(self, double[:] params=?) except -1
    cpdef double logLikelihood(self, double[:] params=?) except? 999.
    cpdef object predict(self, object xTest)
    cpdef object gradient(self, object xTest)
