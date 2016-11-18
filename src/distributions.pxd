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
        double operator()(mTwister generator)
        cppclass param_type:
            param_type(int,double)
        void param(param_type)

cdef class RandomEngine:
    cpdef setSeed(self,unsigned long int i)
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
cpdef double uniformRand(double lower=?, double upper=?, RandomEngine engine=?)
cpdef double uniformPDF(double x, double lower=?, double upper=?)
cpdef double uniformLogPDF(double x, double lower=?, double upper=?)
cpdef double uniformCDF(double x, double lower=?, double upper=?)
cpdef double uniformDLDU(double x, double lower=?, double upper=?)
cpdef double uniformDLDL(double x, double lower=?, double upper=?)
cpdef double uniformMean(double lower=?, double upper=?)
cpdef double uniformVar(double lower=?, double upper=?)
cpdef double uniformStd(double lower=?, double upper=?)


# Normal
cpdef double normalRand(double mean=?, double sigma=?, RandomEngine engine = ?)
cpdef double normalPDF(double x, double mean=?, double sigma=?)
cpdef double normalLogPDF(double x, double mean=?, double sigma=?)
cpdef double normalCDF(double x, double mean=?, double sigma=?)
cpdef double normalDLDM(double x, double mean=?, double sigma=?)
cpdef double normalDLDX(double x, double mean=?, double sigma=?)
cpdef double normalDLDV(double x, double mean=?, double sigma=?)
cpdef double normalDLDS(double x, double mean=?, double sigma=?)
cpdef double normalMean(double mean=?, double sigma=?)
cpdef double normalVar(double mean=?, double sigma=?)
cpdef double normalStd(double mean=?, double sigma=?)
cpdef double normalMode(double mean=?, double sigma=?)

# Multivariate Normal
cpdef mvNormalRand(double[:] mean, double[:,:] covariance, double[:] output)
cpdef mvNormalPDF(double[:] x, double[:] mean, double[:,:] covariance)

# Gamma
cpdef double gammaRand(double shape, double rate, RandomEngine engine = ?)
cpdef double gammaPDF(double x, double shape, double rate)
cpdef double gammaLogPDF(double x, double shape, double rate)
cpdef double gammaCDF(double x, double shape, double rate)
cpdef double gammaDLDA(double x, double shape, double rate)
cpdef double gammaDLDB(double x, double shape, double rate)
cpdef double gammaDLDX(double x, double shape, double rate)
cpdef double gammaMean(double shape, double rate)
cpdef double gammaVar(double shape, double rate)
cpdef double gammaStd(double shape, double rate)
cpdef double gammaMode(double shape, double rate)

# Inverse-Gamma

cpdef double invGammaRand(double shape, double rate, RandomEngine engine=?)
cpdef double invGammaPDF(double x, double shape, double rate)
cpdef double invGammaLogPDF(double x, double shape, double rate)
cpdef double invGammaCDF(double x, double shape, double rate)
cpdef double invGammaDLDA(double x, double shape, double rate)
cpdef double invGammaDLDB(double x, double shape, double rate)
cpdef double invGammaDLDX(double x, double shape, double rate)
cpdef double invGammaMean(double shape, double rate)
cpdef double invGammaVar(double shape, double rate)
cpdef double invGammaStd(double shape, double rate)
cpdef double invGammaMode(double shape, double rate)

# Beta
cpdef double betaRand(double alpha, double beta, RandomEngine engine = ?)
cpdef double betaPDF(double x, double alpha, double beta)
cpdef double betaLogPDF(double x, double alpha, double beta)
cpdef double betaCDF(double x, double alpha, double beta)
cpdef double betaDLDA(double x, double alpha, double beta)
cpdef double betaDLDB(double x, double alpha, double beta)
cpdef double betaMean(double alpha, double beta)
cpdef double betaVar(double alpha, double beta)
cpdef double betaStd(double alpha, double beta)
cpdef double betaMode(double alpha, double beta)

# Poisson
cpdef double poissonRand(double lamb, RandomEngine engine=?)
cpdef double poissonPDF(int x, double lamb)
cpdef double poissonLogPDF(int x, double lamb)
cpdef double poissonCDF(double x, double lamb)
cpdef double poissonDLDL(int x, double lamb)
cpdef double poissonMean(double lamb)
cpdef double poissonVar(double lamb)
cpdef double poissonStd(double lamb)
cpdef int poissonMode(double lamb)

# Exponential
cpdef double exponentialRand(double lamb, RandomEngine engine=?)
cpdef double exponentialPDF(double x, double lamb)
cpdef double exponentialLogPDF(double x, double lamb)
cpdef double exponentialCDF(double x, double lamb)
cpdef double exponentialDLDL(double x, double lamb)
cpdef double exponentialMean(double lamb)
cpdef double exponentialVar(double lamb)
cpdef double exponentialStd(double lamb)
cpdef double exponentialMode(double lamb)

# Binomial
cpdef double binomialRand(int n, double probability, RandomEngine engine=?)
cpdef double binomialPDF(int x, int n, double probability)
cpdef double binomialLogPDF(int x, int n, double probability)
cpdef double binomialCDF(double x, int n, double probability)
cpdef double binomialDLDP(int x, int n, double probability)
cpdef double binomialMean(int n, double probability)
cpdef double binomialVar(int n, double probability)
cpdef double binomialStd(int n, double probability)
cpdef double binomialMode(int n, double probability)
