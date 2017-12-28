cimport cython

# Standard library
from libc.math cimport log, log10, sqrt, exp, sin, cos, tan, acos, asin, atan, atan2, sinh, cosh, tanh, M_PI as pi, INFINITY as inf, NAN as nan, isnan
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
cpdef double incBeta(double x, double a, double b)

# Special functions
cpdef double expit(double x) except +
cpdef double logit(double x) except +

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

    # Output
    cdef readonly object samples
    cdef readonly object results
    cdef public object initialPosition
    cdef double[:,:] sampleView
    cdef public object accepted
    cdef int[:] acceptedView
    cdef Size trials
    cdef vector[Accumulator] sampleStats

    # User Defined Functions
    cdef object pyLogProbability
    cdef int pyLogProbArgNum
    cpdef double logProbability(self, double[:] position, double[:] gradient, bint computeGradient) except +
    cdef void extraInitialization(self)
    cdef void sample(self)

    # User-called functions
    cpdef object run(self, Size nSamples, object x0, Size burnIn=?, Size thinning=?, Size recordStart=?, Size recordStop=?, bint collectStats=?, Size threads=?, bint showProgress=?) except +
    cpdef object getStats(self) except +
    cpdef object getAcceptance(self) except +
    cpdef object testGradient(self, double[:] x0, double eps=?) except +
    cpdef object gradientDescent(self, double[:] x0, double step=?, double eps=?) except +
    cpdef object simulatedAnnealing(self, double[:] x0, Size nSteps=?, Size nQuench=?, double T0=?, double width=?) except +
    cpdef void addMetropolis(self, covariance=?, Size dStart=?, Size dStop=?) except +
    cpdef void addAdaptiveMetropolis(self, covariance=?, int adaptAfter=?, int refreshPeriod=?, double eps=?,  Size dStart=?, Size dStop=?) except +
    cpdef void addHMC(self, Size nSteps, double stepSize, Size dStart=?, Size dStop=?) except +
    cpdef void printSamplers(self) except +
    cpdef void clearSamplers(self) except +
    cpdef SamplerData getSampler(self, unsigned int i=?) except +
    cpdef object getProposalCov(self, unsigned int i=?) except +
    cpdef object summary(self, paramIndices=?, returnString=?) except +

    # Structural functions
    cdef void progressBar(self, Size i, Size N, object header) except +
    cdef void _setMemoryViews_(self) except +
    cdef void record(self,Size i) except +
    cdef void recordStats(self) except +
    cdef void bouncingMove(self, double stepSize, Size dStart, Size dStop) except +
    cdef void onlineCovar(self, double[:] mu, double[:,:] covar, double[:] x, int t, double eps=?) except +

    # Sampling functions
    cdef double hmcStep(self,Size nSteps, double stepSize, Size dStart, Size dStop, double logP0=?) except +
    cdef double metropolisStep(self, Size dStart, Size dStop, double logP0=?) except +
    cdef double adaptiveStep(self, Size dStart, Size dStop, vector[double]* state, vector[int]* idata, double logP0=?) except +
    cdef double metropolisCorrStep(self, Size dStart, Size dStop, double[:,:] proposeChol, double logP0=?) except +
    cdef double[:] regressionStep(self, double[:,:] x1, double[:] y1, double[:] output=?) except +

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

    cpdef object getValues(self) except +
    cpdef object getNPoints(self) except +
    cpdef object getIndices(self) except +
    cpdef object getWeights(self) except +
    cpdef object getStrides(self) except +
    cpdef Size ind(self, Size[:] p) except +
    cpdef bint locatePoints(self, double[:] point) except +
    cpdef double interp(self, double[:] points, double [:] gradient=?, bint locate=?, bint debug=?) except +
    cpdef void bounceMove(self, double[:] x0, double[:] displacement, bint[:] bounced) except +
    cpdef double findEdge(self, Size index, Size dim) except +
    cpdef void interpN(self,double[:,:] points, double[:] output) except +
    cpdef void printInfo(self) except +
