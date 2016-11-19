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

    # Working memory
    cdef object _workingMemory_
    cdef double[:] x
    cdef double[:] xPropose
    cdef double[:] momentum
    cdef double[:] gradient

    # Output
    cdef public object samples
    cdef double[:,:] sampleView
    cdef public double acceptanceRate
    cdef vector[Accumulator] sampleStats

    # User Defined Functions
    cdef object pyLogProbability
    cpdef double logProbability(self, double[:] position, double[:] gradient, bint computeGradient)

    # User-called functions
    cpdef object run(self, Size nSamples, object x0, Size burnIn=?, Size thinning=?, Size recordStart=?, Size recordStop=?, bint collectStats=?, Size threads=?)
    cpdef object getStats(self)
    cpdef void testGradient(self, double[:] x0, double eps=?)
    cpdef object gradientDescent(self, double[:] x0, double step=?, double eps=?)
    cpdef object simulatedAnnealing(self, double[:] x0, Size nSteps=?, Size nQuench=?, double T0=?, double width=?)
    cpdef void addMetropolis(self, Size dStart, Size dStop)
    cpdef void addHMC(self, Size nSteps, double stepSize, Size dStart, Size dStop)
    cpdef void printSamplers(self)
    cpdef void clearSamplers(self)

    # Structural functions
    cdef void _setMemoryViews_(self)
    cdef void sample(self)
    cdef void record(self,Size i)
    cdef void recordStats(self)
    cdef void bouncingMove(self, double stepSize, Size dStart, Size dStop)

    # Sampling functions
    cdef void hmcStep(self,Size nSteps, double stepSize, Size dStart, Size dStop)
    cdef void metropolisStep(self, Size dStart, Size dStop)
    cdef double[:] regressionStep(self, double[:,:] x1, double[:] y1, double[:] output=?)

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
    cpdef Size ind(self, Size[:] p)
    cpdef bint locatePoints(self, double[:] point)
    cpdef double interp(self, double[:] points, double [:] gradient=?, bint locate=?, bint debug=?)
    cpdef void bounceMove(self, double[:] x0, double[:] displacement, bint[:] bounced)
    cpdef double findEdge(self, Size index, Size dim)
    cpdef void interpN(self,double[:,:] points, double[:] output)
    cpdef void printInfo(self)
