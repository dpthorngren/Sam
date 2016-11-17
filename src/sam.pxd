cimport numpy as np
cimport cython
from scipy.stats import multivariate_normal

# Standard library
from libc.math cimport log, log10, sqrt, exp, sin, cos, tan, acos, asin, atan, atan2, sinh, cosh, tanh, M_PI as pi, INFINITY as infinity, NAN as nan, isnan

# Boost special functions
cdef extern from "<boost/math/special_functions.hpp>" namespace "boost::math":
    cpdef double asinh(double x) except +
    cpdef double acosh(double x) except +
    cpdef double atanh(double x) except +
    cpdef double beta(double a, double b) except +
    cdef double _incBeta "boost::math::beta"(double a, double b, double x) except +
    cpdef double gamma "boost::math::tgamma"(double x) except +
    cpdef double digamma(double x) except +
    cpdef double binomial_coefficient[double](unsigned int n, unsigned int k) except +

# Wrapper to fix ordering not matching other functions.
cpdef double incBeta(double x, double a, double b)

# Type definition
ctypedef Py_ssize_t Size

include "distributions.pxd"

cdef class Sam:
    # Parameters
    cdef Size nDim
    cdef double[:] scale
    cdef double[:] upperBoundaries
    cdef double[:] lowerBoundaries

    # Working memory
    cdef double[:] x
    cdef double[:] xPropose
    cdef double[:] momentum
    cdef double[:] gradient

    # Output
    cdef public object samples
    cdef double[:,:] sampleView
    cdef public double acceptanceRate

    # Mandatory user Defined Functions
    cpdef double logProbability(self, double[:] position)
    cpdef void gradLogProbability(self, double[:] position, double[:] output)

    # User-called functions
    cpdef object run(self, Size nSamples, double[:] x0, Size burnIn=?, Size thinning=?)
    cpdef void testGradient(self, double[:] x0, double eps=?)
    cpdef object gradientDescent(self, double[:] x0, double step=?, double eps=?)
    cpdef object simulatedAnnealing(self, double[:] x0, Size nSteps=?, Size nQuench=?, double T0=?, double width=?)

    # Structural functions
    cdef void sample(self)
    cdef void record(self,Size i)
    cdef void bouncingMove(self, double stepSize, Size dMin, Size dMax)

    # Sampling functions
    cdef void hmcStep(self,Size nSteps, double stepSize, Size dMin, Size dMax)
    cdef void metropolisStep(self, double[:] proposalStd, Size dMin, Size dMax)
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

    cpdef Size ind(self, Size[:] p)
    cpdef bint locatePoints(self, double[:] point)
    cpdef double interp(self, double[:] points, double [:] gradient=?, bint locate=?, bint debug=?)
    cpdef void bounceMove(self, double[:] x0, double[:] displacement, bint[:] bounced)
    cpdef double findEdge(self, Size index, Size dim)
    cpdef void interpN(self,double[:,:] points, double[:] output)
    cpdef void printInfo(self)
