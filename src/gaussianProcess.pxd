# Kernel functions and derivatives
cdef double sqExpKernel(double scaledDist)
cdef double sqExpKernelDeriv(double scaledDist)
cdef double expKernel(double scaledDist)
cdef double matern32Kernel(double scaledDist)
cdef double matern32KernelDeriv(double scaledDist)
cdef double matern52Kernel(double scaledDist)
cdef double matern52KernelDeriv(double scaledDist)

# Covariance function builders
cdef int makeCov(double[:,:] x, double[:] params, double[:,:] output, double(*kernel)(double) , double[:,:] xPrime=?, double[:] yErr=?) except -1
# cdef int makeGradientCov(double[:,:] x, double[:] params, double[:,:] output, double(*kernelDeriv)(double), double[:] xPrime) except -1

cdef class GaussianProcess:
    cdef readonly double[:,:] x
    cdef readonly double[:] y
    cdef readonly double[:] yErr
    cdef readonly double[:] params
    cdef readonly double[:,:] covChol
    cdef double[:] alpha
    cdef bint choleskyFresh
    cdef bint alphaFresh
    cdef double (*kernelPtr)(double)
    cdef double (*kernelDerivPtr)(double)
    cdef readonly Size nParams
    cdef readonly Size nDim
    cdef readonly Size nData
    cdef object kernelName

    cpdef int precompute(self, double[:] params=?, bint force=?) except -1
    cpdef double logLikelihood(self, double[:] params=?) except? 999.
    cpdef object predict(self, object xTest, bint getCovariance=?)
    cpdef object draw(self, object xTest, Size nDraws=?)
    cpdef object gradient(self, object xTest)
    cpdef int _gradient_(self, double[:] xTest, double[:] output) except -1
    cpdef object setY(self, object newY)
