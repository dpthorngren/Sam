# Boost special functions
cdef extern from "<boost/math/special_functions.hpp>" namespace "boost::math":
    cdef double asinh(double x) except +
    cdef double acosh(double x) except +
    cdef double atanh(double x) except +
    cdef double beta(double a, double b) except +
    # TODO: Fix incomplete beta function
    # cdef double incBeta "ibetac"(double a, double b, double x) except +
    cdef double gamma "tgamma"(double x) except +
    cdef double digamma(double x) except +

# Boost random variable distribtions (Descriptive functions -- not RNGs)
cdef extern from "<boost/math/distributions.hpp>" namespace "boost::math":
    # Normal
    cdef cppclass normal:
        normal(double mean, double std) except +
    double pdf(normal d, double x) except +
    double cdf(normal d, double x) except +
    # Gamma
    cdef cppclass gamma_info "boost::math::gamma_distribution"[double]:
        gamma_info(double shape, double scale) except +
    double pdf(gamma_info d, double x) except +
    double cdf(gamma_info d, double x) except +
    # Beta
    cdef cppclass beta_info "boost::math::beta_distribution"[double]:
        beta_info(double shape, double scale) except +
    double pdf(beta_info d, double x) except +
    double cdf(beta_info d, double x) except +

# Random number generator
cdef extern from "<boost/random.hpp>":
    # The core generator the other classes use
    cdef cppclass mTwister "boost::mt19937":
        mTwister(int seed) except +
        mTwister() except +
    cdef cppclass normal_rng "boost::normal_distribution"[double]:
        double operator()(mTwister generator)
    cdef cppclass gamma_rng "boost::gamma_distribution"[double]:
        double operator()(mTwister generator)
        cppclass param_type:
            param_type(double, double)
        param_type param(param_type)
cdef extern from "<boost/random/beta_distribution.hpp>":
    cdef cppclass beta_rng "boost::random::beta_distribution"[double]:
        double operator()(mTwister generator)
        cppclass param_type:
            param_type(double, double)
        param_type param(param_type)

# Standard library
from libc.math cimport log, log10, sqrt, exp, sin, cos, tan, acos, asin, atan, atan2, sinh, cosh, tanh, M_PI as pi

ctypedef Py_ssize_t Size

cdef class HMCSampler:
    # Parameters
    cdef Size nDim
    cdef double stepSize

    # Working memory
    cdef double[:] x
    cdef double[:] v
    cdef double[:] xPropose
    cdef double[:] vPropose
    cdef double[:] gradient

    # Output
    cdef object samples

    # Functions
    cpdef double logProbability(self, double[:] position)
    cpdef void gradLogProbability(self, double[:] position, double[:] output)
    cdef void simTrajectory(self, Size nSteps)
    cpdef object run(self, Size nSamples, Size nSteps, double[:] x0, Size burnIn=?)
    cpdef recordTrajectory(self,double[:] x0, double[:] v0, Size nSteps)

# Distribution classes
cdef class _normal:
    cdef mTwister _generator
    cdef normal_rng _rand
    cpdef double pdf(self,double x, double mean=?, double sigma=?)
    cpdef double logPDF(self,double x, double mean=?, double sigma=?)
    cpdef double cdf(self,double x, double mean=?, double sigma=?)
    cpdef double dldm(self,double x, double mean=?, double sigma=?)
    cpdef double dldv(self,double x, double mean=?, double sigma=?)
    cpdef double dlds(self,double x, double mean=?, double sigma=?)
    cpdef double rand(self,double mean=?, double sigma=?)
    cpdef double mean(self, double mean=?, double sigma=?)
    cpdef double var(self, double mean=?, double sigma=?)
    cpdef double std(self, double mean=?, double sigma=?)
    cpdef double mode(self, double mean=?, double sigma=?)

cdef class _gamma:
    cdef mTwister _generator
    cdef gamma_rng _rand
    cpdef double pdf(self,double x, double shape, double rate)
    cpdef double logPDF(self,double x, double shape, double rate)
    cpdef double cdf(self,double x, double shape, double rate)
    cpdef double dlda(self,double x, double shape, double rate)
    cpdef double dldb(self,double x, double shape, double rate)
    cpdef double dldx(self,double x, double shape, double rate)
    cpdef double rand(self,double shape, double rate)
    cpdef double mean(self, double shape, double rate)
    cpdef double var(self, double shape, double rate)
    cpdef double std(self, double shape, double rate)
    cpdef double mode(self, double shape, double rate)

cdef class _invGamma:
    cdef mTwister _generator
    cdef gamma_rng _rand
    cpdef double pdf(self,double x, double shape, double rate)
    cpdef double logPDF(self,double x, double shape, double rate)
    cpdef double cdf(self,double x, double shape, double rate)
    cpdef double dlda(self,double x, double shape, double rate)
    cpdef double dldb(self,double x, double shape, double rate)
    cpdef double dldx(self,double x, double shape, double rate)
    cpdef double rand(self,double shape, double rate)
    cpdef double mean(self, double shape, double rate)
    cpdef double var(self, double shape, double rate)
    cpdef double std(self, double shape, double rate)
    cpdef double mode(self, double shape, double rate)

cdef class _beta:
    cdef mTwister _generator
    cdef beta_rng _rand
    cpdef double pdf(self,double x, double alpha, double beta)
    cpdef double logPDF(self,double x, double alpha, double beta)
    cpdef double cdf(self,double x, double alpha, double beta)
    cpdef double dlda(self,double x, double alpha, double beta)
    cpdef double dldb(self,double x, double alpha, double beta)
    cpdef double rand(self,double alpha, double beta)
    cpdef double mean(self, double alpha, double beta)
    cpdef double var(self, double alpha, double beta)
    cpdef double std(self, double alpha, double beta)
    cpdef double mode(self, double alpha, double beta)
