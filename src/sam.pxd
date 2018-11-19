cimport cython

# BLAS and LAPACK
cimport scipy.linalg.cython_lapack as lapack
cimport scipy.linalg.cython_blas as blas

# Standard library
from libc.math cimport log, log10, abs, sqrt, exp, sin, cos, tan, acos, asin, atan, atan2, sinh, cosh, tanh, M_PI as pi, INFINITY as inf, NAN as nan, isnan
from libcpp.vector cimport vector

# Boost library (RNG functions declared separately in distributions.pxd)
cdef extern from "<boost/math/special_functions.hpp>" namespace "boost::math":
    cpdef double asinh(double x) except +
    cpdef double acosh(double x) except +
    cpdef double atanh(double x) except +
    cpdef double beta(double a, double b) except +
    cdef double _incBeta "boost::math::beta"(double a, double b, double x) except +
    cpdef double gamma "boost::math::tgamma"(double x) except +
    cpdef double digamma(double x) except +
    cpdef double binomial_coefficient[double](unsigned int n, unsigned int k) except +
cdef extern from "<boost/accumulators/statistics/stats.hpp>":
    pass
cdef extern from "<boost/accumulators/statistics/mean.hpp>" namespace "boost::accumulators":
    double mean(Accumulator)
cdef extern from "<boost/accumulators/statistics/variance.hpp>" namespace "boost::accumulators":
    double variance(Accumulator)
cdef extern from "<boost/accumulators/accumulators.hpp>":
    cdef cppclass Accumulator u"boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::mean, boost::accumulators::tag::variance> > ":
        void operator()(double)

# Wrapper to fix ordering not matching other functions.
cpdef double incBeta(double x, double a, double b) except -1.

# Special functions
cpdef double expit(double x) except -1.
cpdef double logit(double x) except? -1.

# Helper functions
cpdef int choleskyInplace(double[:,:] x) except -1
cdef int gpKernel(double[:,:] x, double[:] params, double[:,:] output, double(*kernel)(double) , double[:,:] xPrime=?) except -1
cdef double gpSqExpCovariance(double scaledDist)
cdef double gpExpCovariance(double scaledDist)
cdef double matern32(double scaledDist)
cdef double matern52(double scaledDist)

# Type definitions
ctypedef Py_ssize_t Size
cdef struct SamplerData:
    # Stores information for each sampler in the samplers list.
    #
    # The user should never need to interact with this struct directly, instead
    # using the add* functions, clearSamplers(), and printSamplers() functions.
    # This struct exists (as opposed to a class) because more logical methods
    # of storage are either not pure c++ (and I'd really like this to be fast)
    # or they are somehow not cython compatible.
    #
    # The sampler type is specified according to the following codes below in
    # parentheses.  dStart and dStop indicate the first and last+1 parameter
    # index that the sampler is to be applied to.  Additional data is stored
    # in idata (for ints) and ddata (for doubles), and varies by the type of
    # sampler, according to the folloing layout:
    #
    # Diagonal Metropolis (0):
    #     idata: empty
    #     ddata: empty
    # HMC (1):
    #     idata[0]: number of steps
    #     ddata[0]: step size
    # Metropolis (2):
    #     idata: empty
    #     ddata[0:n**2]: flattened proposal covariance matrix cholesky
    # adaptiveMetropolis (3):
    #     idata[0]: number of samples used so far to produce proposal covariance
    #     idata[1]: number of samples to collect before using adapted covariance
    #     idata[2]: samples between recomputation of the proposal covariance
    #     ddata[0]: epsilon (added diagonal component of the proposal covariance)
    #     ddata[1:1+n]: mean of adaptive samples
    #     ddata[1+n:1+n+n**2]: covariance of adaptive samples
    #     ddata[1+n+n**2:1+n+2*n**2]: current proposal cholesky
    Size samplerType, dStart, dStop
    vector[int] idata
    vector[double] ddata

include "distributions.pxd"

cdef class Sam:
    # Parameters
    cdef Size nDim, nSamples
    cdef Size burnIn, thinning
    cdef Size recordStart, recordStop
    cdef bint collectStats
    cdef bint readyToRun
    cdef vector[SamplerData] samplers
    cdef double[:] scale
    cdef double[:] upperBoundaries
    cdef double[:] lowerBoundaries
    cdef bint hasBoundaries
    cdef bint showProgress

    # Working memory
    cdef object _workingMemory_
    cdef double[:] x
    cdef double[:] xPropose
    cdef double[:] momentum
    cdef double[:] gradient
    cdef double lastLogProb

    # Output
    cdef readonly object samples
    cdef readonly object results
    cdef readonly object samplesLogProb
    cdef readonly object resultsLogProb
    cdef public object initialPosition
    cdef double[:,:] sampleView
    cdef double[:] samplesLogProbView
    cdef public object accepted
    cdef int[:] acceptedView
    cdef Size trials
    cdef vector[Accumulator] sampleStats

    # User Defined Functions
    cdef object pyLogProbability
    cdef int pyLogProbArgNum
    cpdef double logProbability(self, double[:] position, double[:] gradient, bint computeGradient) except? 999.
    cdef int extraInitialization(self) except -1
    cdef int sample(self) except -1

    # User-called functions
    cpdef object run(self, Size nSamples, x, Size burnIn=?, Size thinning=?, Size recordStart=?, Size recordStop=?, bint collectStats=?, Size threads=?, bint showProgress=?)
    cpdef object getStats(self)
    cpdef object getAcceptance(self)
    cpdef object getBIC(self,prior,nPoints)
    cpdef object getAIC(self,prior)
    cpdef object getDIC(self,prior)
    cpdef object testGradient(self, x, double eps=?)
    cpdef object gradientDescent(self, x, double step=?, double eps=?)
    cpdef object simulatedAnnealing(self, x, Size nSteps=?, Size nQuench=?, double T0=?, double width=?)
    cpdef object addMetropolis(self, covariance=?, Size dStart=?, Size dStop=?)
    cpdef object addAdaptiveMetropolis(self, covariance=?, int adaptAfter=?, int recordAfter=?, int refreshPeriod=?, double scaling=?, double eps=?,  Size dStart=?, Size dStop=?)
    cpdef object addHMC(self, Size nSteps, double stepSize, Size dStart=?, Size dStop=?)
    cpdef object printSamplers(self)
    cpdef object clearSamplers(self)
    cpdef object getSampler(self, unsigned int i=?)
    cpdef object getProposalCov(self, unsigned int i=?)
    cpdef object summary(self, paramIndices=?, returnString=?)

    # Structural functions
    cdef int progressBar(self, Size i, Size N, object header) except -1
    cdef int _setMemoryViews_(self) except -1
    cdef int record(self,Size i) except -1
    cdef int recordStats(self) except -1
    cdef int bouncingMove(self, double stepSize, Size dStart, Size dStop) except -1
    cdef int onlineCovar(self, double[:] mu, double[:,:] covar, double[:] x, int t, double scaling, double eps=?) except -1

    # Sampling functions
    cdef double hmcStep(self,Size nSteps, double stepSize, Size dStart, Size dStop, double logP0=?) except 999.
    cdef double metropolisStep(self, Size dStart, Size dStop, double logP0=?) except 999.
    cdef double adaptiveStep(self, Size dStart, Size dStop, vector[double]* state, vector[int]* idata, double logP0=?) except 999.
    cdef double metropolisCorrStep(self, Size dStart, Size dStop, double[:,:] proposeChol, double logP0=?) except 999.
    cdef double[:] regressionStep(self, double[:,:] x1, double[:] y1, double[:] output=?) except *

# Griddy
cdef class Griddy:
    # Grid information
    cdef double[:,:] axes
    cdef double[:] values
    cdef Size[:] nPoints
    cdef Size[:] strides
    cdef Size nDim

    # Working memory
    cdef double[:] weights
    cdef double[:] widths
    cdef Size[:] indices
    cdef Size[:] tempIndices

    cpdef object getValues(self)
    cpdef object getNPoints(self)
    cpdef object getIndices(self)
    cpdef object getWeights(self)
    cpdef object getStrides(self)
    cpdef Size ind(self, Size[:] p) except -1
    cpdef bint locatePoints(self, double[:] point) except? True
    cpdef double interp(self, double[:] points, double [:] gradient=?, bint locate=?, bint debug=?) except? -1.
    cpdef int bounceMove(self, double[:] x0, double[:] displacement, bint[:] bounced) except -1
    cpdef double findEdge(self, Size index, Size dim) except? 999.
    cpdef int interpN(self,double[:,:] points, double[:] output) except -1
    cpdef object printInfo(self)
