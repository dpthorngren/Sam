# Boost random variable distribtions (Descriptive functions -- not RNGs)
cdef extern from "<boost/math/distributions.hpp>" namespace "boost::math":
    # Normal
    cdef cppclass normal_info "boost::math::normal":
        normal_info(double mean, double std) except +
    double pdf(normal_info d, double x) except +
    double cdf(normal_info d, double x) except +
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

# Random number generators
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
        int operator()(mTwister generator)
        cppclass param_type:
            param_type(int,double)
        void param(param_type)

cdef class RandomEngine:
    cpdef object setSeed(self,unsigned long int i)
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
cpdef double uniformRand(double lower=?, double upper=?, RandomEngine engine=?) except? 999.
cpdef double uniformPDF(double x, double lower=?, double upper=?) except? 999.
cpdef double uniformLogPDF(double x, double lower=?, double upper=?) except? 999.
cpdef double uniformCDF(double x, double lower=?, double upper=?) except? 999.
cpdef double uniformDLDU(double x, double lower=?, double upper=?) except? 999.
cpdef double uniformDLDL(double x, double lower=?, double upper=?) except? 999.
cpdef double uniformMean(double lower=?, double upper=?) except? 999.
cpdef double uniformVar(double lower=?, double upper=?) except? 999.
cpdef double uniformStd(double lower=?, double upper=?) except? 999.


# Normal
cpdef double normalRand(double mean=?, double sigma=?, RandomEngine engine = ?) except? 999.
cpdef double normalPDF(double x, double mean=?, double sigma=?) except? 999.
cpdef double normalLogPDF(double x, double mean=?, double sigma=?) except? 999.
cpdef double normalCDF(double x, double mean=?, double sigma=?) except? 999.
cpdef double normalDLDM(double x, double mean=?, double sigma=?) except? 999.
cpdef double normalDLDX(double x, double mean=?, double sigma=?) except? 999.
cpdef double normalDLDV(double x, double mean=?, double sigma=?) except? 999.
cpdef double normalDLDS(double x, double mean=?, double sigma=?) except? 999.
cpdef double normalMean(double mean=?, double sigma=?) except? 999.
cpdef double normalVar(double mean=?, double sigma=?) except? 999.
cpdef double normalStd(double mean=?, double sigma=?) except? 999.
cpdef double normalMode(double mean=?, double sigma=?) except? 999.

# Multivariate Normal
cpdef double[:] mvNormalRand(double[:] mean, double[:,:] covariance, double[:] output=?, bint isChol=?, RandomEngine engine=?) except *
cpdef double mvNormalPDF(double[:] x, double[:] mean, double[:,:] covariance, bint isChol=?) except? 999.
cpdef double mvNormalLogPDF(double[:] x, double[:] mean, double[:,:] covariance, bint isChol=?) except? 999.

# Gamma
cpdef double gammaRand(double shape, double rate, RandomEngine engine = ?) except? 999.
cpdef double gammaPDF(double x, double shape, double rate) except? 999.
cpdef double gammaLogPDF(double x, double shape, double rate) except? 999.
cpdef double gammaCDF(double x, double shape, double rate) except? 999.
cpdef double gammaDLDA(double x, double shape, double rate) except? 999.
cpdef double gammaDLDB(double x, double shape, double rate) except? 999.
cpdef double gammaDLDX(double x, double shape, double rate) except? 999.
cpdef double gammaMean(double shape, double rate) except? 999.
cpdef double gammaVar(double shape, double rate) except? 999.
cpdef double gammaStd(double shape, double rate) except? 999.
cpdef double gammaMode(double shape, double rate) except? 999.

# Inverse-Gamma
cpdef double invGammaRand(double shape, double rate, RandomEngine engine=?) except? 999.
cpdef double invGammaPDF(double x, double shape, double rate) except? 999.
cpdef double invGammaLogPDF(double x, double shape, double rate) except? 999.
cpdef double invGammaCDF(double x, double shape, double rate) except? 999.
cpdef double invGammaDLDA(double x, double shape, double rate) except? 999.
cpdef double invGammaDLDB(double x, double shape, double rate) except? 999.
cpdef double invGammaDLDX(double x, double shape, double rate) except? 999.
cpdef double invGammaMean(double shape, double rate) except? 999.
cpdef double invGammaVar(double shape, double rate) except? 999.
cpdef double invGammaStd(double shape, double rate) except? 999.
cpdef double invGammaMode(double shape, double rate) except? 999.

# Beta
cpdef double betaRand(double alpha, double beta, RandomEngine engine = ?) except? 999.
cpdef double betaPDF(double x, double alpha, double beta) except? 999.
cpdef double betaLogPDF(double x, double alpha, double beta) except? 999.
cpdef double betaCDF(double x, double alpha, double beta) except? 999.
cpdef double betaDLDA(double x, double alpha, double beta) except? 999.
cpdef double betaDLDB(double x, double alpha, double beta) except? 999.
cpdef double betaMean(double alpha, double beta) except? 999.
cpdef double betaVar(double alpha, double beta) except? 999.
cpdef double betaStd(double alpha, double beta) except? 999.
cpdef double betaMode(double alpha, double beta) except? 999.

# Poisson
cpdef double poissonRand(double lamb, RandomEngine engine=?) except? 999.
cpdef double poissonPDF(int x, double lamb) except? 999.
cpdef double poissonLogPDF(int x, double lamb) except? 999.
cpdef double poissonCDF(double x, double lamb) except? 999.
cpdef double poissonDLDL(int x, double lamb) except? 999.
cpdef double poissonMean(double lamb) except? 999.
cpdef double poissonVar(double lamb) except? 999.
cpdef double poissonStd(double lamb) except? 999.
cpdef int poissonMode(double lamb) except? -1

# Exponential
cpdef double exponentialRand(double lamb, RandomEngine engine=?) except? 999.
cpdef double exponentialPDF(double x, double lamb) except? 999.
cpdef double exponentialLogPDF(double x, double lamb) except? 999.
cpdef double exponentialCDF(double x, double lamb) except? 999.
cpdef double exponentialDLDL(double x, double lamb) except? 999.
cpdef double exponentialMean(double lamb) except? 999.
cpdef double exponentialVar(double lamb) except? 999.
cpdef double exponentialStd(double lamb) except? 999.
cpdef double exponentialMode(double lamb) except? 999.

# Binomial
cpdef int binomialRand(int n, double probability, RandomEngine engine=?) except? -1
cpdef double binomialPDF(int x, int n, double probability) except? 999.
cpdef double binomialLogPDF(int x, int n, double probability) except? 999.
cpdef double binomialCDF(double x, int n, double probability) except? 999.
cpdef double binomialDLDP(int x, int n, double probability) except? 999.
cpdef double binomialMean(int n, double probability) except? 999.
cpdef double binomialVar(int n, double probability) except? 999.
cpdef double binomialStd(int n, double probability) except? 999.
cpdef double binomialMode(int n, double probability) except? 999.
