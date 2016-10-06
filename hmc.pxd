# Type definition
ctypedef Py_ssize_t Size

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
    # cdef double binomial_coefficient[double](unsigned int n, unsigned int k) except +

# Boost random variable distribtions (Descriptive functions -- not RNGs)
cdef extern from "<boost/math/distributions.hpp>" namespace "boost::math":
    # Normal
    # TODO: Rename
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
    # Poisson
    cdef cppclass pois_info "boost::math::poisson_distribution"[double]:
        pois_info(double lamb) except +
    double pdf(pois_info d, double x) except +
    double cdf(pois_info d, double x) except +
    # Binomial
    cdef cppclass binom_info "boost::math::binomial_distribution"[double]:
        binom_info(int n, double probability) except +
    double pdf(binom_info d, int x) except +
    double cdf(binom_info d, double x) except +

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
        void param(param_type)
cdef extern from "<boost/random/uniform_01.hpp>":
    cdef cppclass uniform_rng "boost::uniform_01"[double]:
        double operator()(mTwister generator)
cdef extern from "<boost/random/beta_distribution.hpp>":
    cdef cppclass beta_rng "boost::random::beta_distribution"[double]:
        double operator()(mTwister generator)
        cppclass param_type:
            param_type(double, double)
        void param(param_type)
cdef extern from "<boost/random/poisson_distribution.hpp>":
    cdef cppclass pois_rng "boost::random::poisson_distribution"[int]:
        double operator()(mTwister generator)
        cppclass param_type:
            param_type(double)
        void param(param_type)
cdef extern from "<boost/random/exponential_distribution.hpp>":
    cdef cppclass expon_rng "boost::random::exponential_distribution"[double]:
        double operator()(mTwister generator)
        cppclass param_type:
            param_type(double)
        void param(param_type)
cdef extern from "<boost/random/binomial_distribution.hpp>":
    cdef cppclass binom_rng "boost::random::binomial_distribution"[int,double]:
        double operator()(mTwister generator)
        cppclass param_type:
            param_type(int,double)
        void param(param_type)

# Standard library
from libc.math cimport log, log10, sqrt, exp, sin, cos, tan, acos, asin, atan, atan2, sinh, cosh, tanh, M_PI as pi, INFINITY as infinity

cdef class HMCSampler:
    # Parameters
    cdef Size nDim
    cdef int _testMode;
    cdef int[:] samplerChoice
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
    cdef void bouncingMove(self, double stepSize, int ID)

    # Sampling functions
    cdef void hmcStep(self,Size nSteps, double stepSize, int ID=?)
    cdef void metropolisStep(self, double[:] proposalStd, int ID=?)
    cdef double[:] regressionStep(self, double[:,:] x1, double[:] y1, double[:] output=?)

# Distribution classes
cdef class RandomEngine:
    # RNG
    cdef mTwister source
    cdef uniform_rng uniform_rand
    cdef normal_rng normal_rand
    cdef gamma_rng gamma_rand
    cdef beta_rng beta_rand
    cdef pois_rng poisson_rand
    cdef expon_rng exponential_rand
    cdef binom_rng binomial_rand

cdef RandomEngine defaultEngine

# Uniform Distribution
cpdef double UniformRand(double lower=?, double upper=?, RandomEngine engine=?)
cpdef double UniformPDF(double x, double lower=?, double upper=?)
cpdef double UniformLogPDF(double x, double lower=?, double upper=?)
cpdef double UniformCDF(double x, double lower=?, double upper=?)
cpdef double UniformDLDU(double x, double lower=?, double upper=?)
cpdef double UniformDLDL(double x, double lower=?, double upper=?)
cpdef double UniformMean(double lower=?, double upper=?)
cpdef double UniformVar(double lower=?, double upper=?)
cpdef double UniformStd(double lower=?, double upper=?)


# Normal
cpdef double NormalRand(double mean=?, double sigma=?, RandomEngine engine = ?)
cpdef double NormalPDF(double x, double mean=?, double sigma=?)
cpdef double NormalLogPDF(double x, double mean=?, double sigma=?)
cpdef double NormalCDF(double x, double mean=?, double sigma=?)
cpdef double NormalDLDM(double x, double mean=?, double sigma=?)
cpdef double NormalDLDX(double x, double mean=?, double sigma=?)
cpdef double NormalDLDV(double x, double mean=?, double sigma=?)
cpdef double NormalDLDS(double x, double mean=?, double sigma=?)
cpdef double NormalMean(double mean=?, double sigma=?)
cpdef double NormalVar(double mean=?, double sigma=?)
cpdef double NormalStd(double mean=?, double sigma=?)
cpdef double NormalMode(double mean=?, double sigma=?)

# Multivariate Normal
cpdef mvNormalRand(double[:] mean, double[:,:] covariance, double[:] output)
cpdef mvNormalPDF(double[:] x, double[:] mean, double[:,:] covariance)

# Gamma
cpdef double GammaRand(double shape, double rate, RandomEngine engine = ?)
cpdef double GammaPDF(double x, double shape, double rate)
cpdef double GammaLogPDF(double x, double shape, double rate)
cpdef double GammaCDF(double x, double shape, double rate)
cpdef double GammaDLDA(double x, double shape, double rate)
cpdef double GammaDLDB(double x, double shape, double rate)
cpdef double GammaDLDX(double x, double shape, double rate)
cpdef double GammaMean(double shape, double rate)
cpdef double GammaVar(double shape, double rate)
cpdef double GammaStd(double shape, double rate)
cpdef double GammaMode(double shape, double rate)

# Inverse-Gamma

cpdef double InvGammaRand(double shape, double rate, RandomEngine engine=?)
cpdef double InvGammaPDF(double x, double shape, double rate)
cpdef double InvGammaLogPDF(double x, double shape, double rate)
cpdef double InvGammaCDF(double x, double shape, double rate)
cpdef double InvGammaDLDA(double x, double shape, double rate)
cpdef double InvGammaDLDB(double x, double shape, double rate)
cpdef double InvGammaDLDX(double x, double shape, double rate)
cpdef double InvGammaMean(double shape, double rate)
cpdef double InvGammaVar(double shape, double rate)
cpdef double InvGammaStd(double shape, double rate)
cpdef double InvGammaMode(double shape, double rate)

# Beta
cpdef double BetaRand(double alpha, double beta, RandomEngine engine = ?)
cpdef double BetaPDF(double x, double alpha, double beta)
cpdef double BetaLogPDF(double x, double alpha, double beta)
cpdef double BetaCDF(double x, double alpha, double beta)
cpdef double BetaDLDA(double x, double alpha, double beta)
cpdef double BetaDLDB(double x, double alpha, double beta)
cpdef double BetaMean(double alpha, double beta)
cpdef double BetaVar(double alpha, double beta)
cpdef double BetaStd(double alpha, double beta)
cpdef double BetaMode(double alpha, double beta)

# Poisson
cpdef double PoissonRand(double lamb, RandomEngine engine=?)
cpdef double PoissonPDF(int x, double lamb)
cpdef double PoissonLogPDF(int x, double lamb)
cpdef double PoissonCDF(double x, double lamb)
cpdef double PoissonDLDL(int x, double lamb)
cpdef double PoissonMean(double lamb)
cpdef double PoissonVar(double lamb)
cpdef double PoissonStd(double lamb)
cpdef int PoissonMode(double lamb)

# Exponential
cpdef double ExponentialRand(double lamb, RandomEngine engine=?)
cpdef double ExponentialPDF(double x, double lamb)
cpdef double ExponentialLogPDF(double x, double lamb)
cpdef double ExponentialCDF(double x, double lamb)
cpdef double ExponentialDLDL(double x, double lamb)
cpdef double ExponentialMean(double lamb)
cpdef double ExponentialVar(double lamb)
cpdef double ExponentialStd(double lamb)
cpdef double ExponentialMode(double lamb)

# Binomial
cpdef double BinomialRand(int n, double probability, RandomEngine engine=?)
cpdef double BinomialPDF(int x, int n, double probability)
cpdef double BinomialLogPDF(int x, int n, double probability)
cpdef double BinomialCDF(double x, int n, double probability)
cpdef double BinomialDLDP(int x, int n, double probability)
cpdef double BinomialMean(int n, double probability)
cpdef double BinomialVar(int n, double probability)
cpdef double BinomialStd(int n, double probability)
cpdef double BinomialMode(int n, double probability)
