cimport numpy as np
cimport cython
from scipy.stats import multivariate_normal

# Standard library
from libc.math cimport log, log10, sqrt, exp, sin, cos, tan, acos, asin, atan, atan2, sinh, cosh, tanh, M_PI as pi, INFINITY as infinity, NAN as nan, isnan
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

# Type definitions
ctypedef Py_ssize_t Size
cdef struct SamplerData:
    # 0 = metropolis, 1 = HMC
    int samplerType
    Size dStart, dStop, nSteps
    double stepSize

include "distributions.pxd"

# Helper functions
cpdef double getWAIC(logLike, samples)
cpdef double getAIC(loglike, samples)
cpdef double getBIC(loglike, samples, nPoints)

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

    # Working memory
    cdef object _workingMemory_
    cdef double[:] x
    cdef double[:] xPropose
    cdef double[:] momentum
    cdef double[:] gradient

    # Output
    cdef public object samples
    cdef double[:,:] sampleView
    cdef public object accepted
    cdef int[:] acceptedView
    cdef Size trials
    cdef vector[Accumulator] sampleStats

    # User Defined Functions
    cdef object pyLogProbability
    cpdef double logProbability(self, double[:] position, double[:] gradient, bint computeGradient)
    cdef void extraInitialization(self)
    cdef void sample(self)

    # User-called functions
    cpdef object run(self, Size nSamples, object x0, Size burnIn=?, Size thinning=?, Size recordStart=?, Size recordStop=?, bint collectStats=?, Size threads=?) except +
    cpdef object getStats(self) except +
    cpdef object getAcceptance(self) except +
    cpdef object testGradient(self, double[:] x0, double eps=?) except +
    cpdef object gradientDescent(self, double[:] x0, double step=?, double eps=?) except +
    cpdef object simulatedAnnealing(self, double[:] x0, Size nSteps=?, Size nQuench=?, double T0=?, double width=?) except +
    cpdef void addMetropolis(self, Size dStart, Size dStop) except +
    cpdef void addHMC(self, Size nSteps, double stepSize, Size dStart, Size dStop) except +
    cpdef void printSamplers(self) except +
    cpdef void clearSamplers(self) except +

    # Structural functions
    cdef void _setMemoryViews_(self) except +
    cdef void record(self,Size i) except +
    cdef void recordStats(self) except +
    cdef void bouncingMove(self, double stepSize, Size dStart, Size dStop) except +

    # Sampling functions
    cdef double hmcStep(self,Size nSteps, double stepSize, Size dStart, Size dStop, double logP0=?) except +
    cdef double metropolisStep(self, Size dStart, Size dStop, double logP0=?) except +
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
