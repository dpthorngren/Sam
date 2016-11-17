import time
cdef class RandomEngine:
    def __init__(self,unsigned long int i):
        self.source = mTwister(i)
defaultEngine = RandomEngine(<unsigned long int>(1000*time.time()))

# ===== Uniform Distribution =====

cpdef double UniformRand(double lower=0, double upper=1, RandomEngine engine=defaultEngine):
    return engine.uniform_rand(engine.source)*(upper-lower) + lower

cpdef double UniformPDF(double x, double lower=0, double upper=1):
        if x > lower and x < upper:
            return 1./(upper-lower)
        return 0.

cpdef double UniformLogPDF(double x, double lower=0, double upper=1):
    return log(UniformPDF(x,lower,upper))

cpdef double UniformCDF(double x, double lower=0, double upper=1):
    if x < lower:
        return 0.
    if x > upper:
        return 1.
    return (x-lower)/(upper-lower)

cpdef double UniformDLDU(double x, double lower=0, double upper=1):
    # TODO: Fix
    return 0.

cpdef double UniformDLDL(double x, double lower=0, double upper=1):
    # TODO: Fix
    return 0.

cpdef double UniformMean(double lower=0, double upper=1):
    return (upper+lower)/2.

cpdef double UniformVar(double lower=0, double upper=1):
    return (upper-lower)**2 / 12.

cpdef double UniformStd(double lower=0, double upper=1):
    return (upper-lower)/sqrt(12.)

# ===== Normal Distribution =====

cpdef double NormalRand(double mean=0, double sigma=1, RandomEngine engine = defaultEngine):
    return engine.normal_rand(engine.source)*sigma + mean

cpdef double NormalPDF(double x, double mean=0, double sigma=1):
    return exp(-(x-mean)**2/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma)

cpdef double NormalLogPDF(double x, double mean=0, double sigma=1):
    return -(x-mean)**2/(2*sigma*sigma) - .5*log(2*pi*sigma*sigma)

cpdef double NormalCDF(double x, double mean=0, double sigma=1):
    return cdf(normal(mean,sigma),x)

cpdef double NormalDLDM(double x, double mean=0, double sigma=1):
    return (x-mean)/(sigma*sigma)

cpdef double NormalDLDX(double x, double mean=0, double sigma=1):
    return (mean-x)/(sigma*sigma)

cpdef double NormalDLDV(double x, double mean=0, double sigma=1):
    return (x-mean)**2 / (2*sigma**4) - .5/sigma**2

cpdef double NormalDLDS(double x, double mean=0, double sigma=1):
    return (x-mean)**2 / sigma**3 - 1./sigma

cpdef double NormalMean(double mean=0, double sigma=1):
    return mean

cpdef double NormalVar(double mean=0, double sigma=1):
    return sigma*sigma

cpdef double NormalStd(double mean=0, double sigma=1):
    return sigma

cpdef double NormalMode(double mean=0, double sigma=1):
    return mean

# ===== Multivariate Normal Distribution =====
#TODO: Re-implement this prototype without using Numpy
cpdef mvNormalRand(double[:] mean, double[:,:] covariance, double[:] output):
    output = np.asarray(mean) + np.linalg.cholesky(covariance)*np.random.randn(mean.shape[0])
    return np.asarray(output).copy()

cpdef mvNormalPDF(double[:] x, double[:] mean, double[:,:] covariance):
    cov = np.asmatrix(covariance)
    offset = np.asmatrix(np.asarray(x)-np.asarray(mean))
    cdef int dim = mean.shape[0]
    return exp(-offset*cov*offset.T/2.0)/\
            ((2.*pi)**(dim/2.) * sqrt(np.linalg.det(cov)))

# ===== Gamma Distribution =====

cpdef double GammaRand(double shape, double rate, RandomEngine engine = defaultEngine):
    engine.gamma_rand.param(gamma_rng.param_type(shape,1./rate))
    return engine.gamma_rand(engine.source)

cpdef double GammaPDF(double x, double shape, double rate):
    cdef gamma_info* temp = new gamma_info(shape,1./rate)
    cdef double return_value = pdf(temp[0],x)
    del temp
    return return_value

cpdef double GammaLogPDF(double x, double shape, double rate):
    return log(GammaPDF(x,shape,rate))

cpdef double GammaCDF(double x, double shape, double rate):
    cdef gamma_info* temp = new gamma_info(shape,1./rate)
    cdef double return_value = cdf(temp[0],x)
    del temp
    return return_value

cpdef double GammaDLDA(double x, double shape, double rate):
    return log(rate) + log(x) - digamma(shape)

cpdef double GammaDLDB(double x, double shape, double rate):
    return shape/rate - x

cpdef double GammaDLDX(double x, double shape, double rate):
    return (shape-1)/x - rate

cpdef double GammaMean(double shape, double rate):
    return shape/rate

cpdef double GammaVar(double shape, double rate):
    return shape/rate**2

cpdef double GammaStd(double shape, double rate):
    return sqrt(shape)/rate

cpdef double GammaMode(double shape, double rate):
    if shape < 1:
        return 0
    return (shape-1)/rate

# ===== Inverse-Gamma Distribution =====

cpdef double InvGammaRand(double shape, double rate, RandomEngine engine=defaultEngine):
    engine.gamma_rand.param(gamma_rng.param_type(shape,1./rate))
    return 1.0/engine.gamma_rand(engine.source)

cpdef double InvGammaPDF(double x, double shape, double rate):
    cdef gamma_info* temp = new gamma_info(shape,rate)
    cdef double return_value = pdf(temp[0],1./x)
    del temp
    return return_value

cpdef double InvGammaLogPDF(double x, double shape, double rate):
    return log(InvGammaPDF(x,shape,rate))

cpdef double InvGammaCDF(double x, double shape, double rate):
    cdef gamma_info* temp = new gamma_info(shape,rate)
    cdef double return_value = cdf(temp[0],1./x)
    del temp
    return return_value

cpdef double InvGammaDLDA(double x, double shape, double rate):
    return log(rate) - log(x) - digamma(shape)

cpdef double InvGammaDLDB(double x, double shape, double rate):
    return shape/rate - 1./x

cpdef double InvGammaDLDX(double x, double shape, double rate):
    return 2.*rate/x - (shape + 1.)/x

cpdef double InvGammaMean(double shape, double rate):
    return rate / (shape - 1)

cpdef double InvGammaVar(double shape, double rate):
    return rate**2  / ((shape-1)**2 * (shape-2))

cpdef double InvGammaStd(double shape, double rate):
    if shape <= 2:
        # TODO: make nan
        return 0
    return rate / ((shape-1)*sqrt(shape-2))

cpdef double InvGammaMode(double shape, double rate):
    return rate/(shape+1.)

# ===== Beta Distribution =====

cpdef double BetaRand(double alpha, double beta, RandomEngine engine = defaultEngine):
    engine.beta_rand.param(beta_rng.param_type(alpha,beta))
    return engine.beta_rand(engine.source)

cpdef double BetaPDF(double x, double alpha, double beta):
    cdef beta_info* temp = new beta_info(alpha,beta)
    cdef double return_value = pdf(temp[0],x)
    del temp
    return return_value

cpdef double BetaLogPDF(double x, double alpha, double beta):
    return log(BetaPDF(x,alpha, beta))

cpdef double BetaCDF(double x, double alpha, double beta):
    cdef beta_info* temp = new beta_info(alpha,beta)
    cdef double return_value = cdf(temp[0],x)
    del temp
    return return_value

cpdef double BetaDLDA(double x, double alpha, double beta):
    return log(x) + digamma(alpha+beta) - digamma(alpha)

cpdef double BetaDLDB(double x, double alpha, double beta):
    return log(1-x) + digamma(alpha+beta) - digamma(alpha)

cpdef double BetaMean(double alpha, double beta):
    return alpha / (alpha+beta)

cpdef double BetaVar(double alpha, double beta):
    return alpha*beta / ((alpha+beta)**2*(alpha+beta+1))

cpdef double BetaStd(double alpha, double beta):
    return sqrt(BetaVar(alpha,beta))

cpdef double BetaMode(double alpha, double beta):
    if alpha <= 1 or beta <= 1.:
        if alpha < beta:
            return 0
        elif alpha > beta:
            return 1
        else:
            return .5
    return (alpha - 1) / (alpha + beta - 2)

# ===== Poisson Distribution =====

cpdef double PoissonRand(double lamb, RandomEngine engine=defaultEngine):
    engine.poisson_rand.param(pois_rng.param_type(lamb))
    return engine.poisson_rand(engine.source)

cpdef double PoissonPDF(int x, double lamb):
    cdef pois_info* temp = new pois_info(lamb)
    cdef double return_value = pdf(temp[0],x)
    del temp
    return return_value

cpdef double PoissonLogPDF(int x, double lamb):
    return log(PoissonPDF(x,lamb))

cpdef double PoissonCDF(double x, double lamb):
    cdef pois_info* temp = new pois_info(lamb)
    cdef double return_value = cdf(temp[0],<int>x)
    del temp
    return return_value

cpdef double PoissonDLDL(int x, double lamb):
    return x/lamb - 1.

cpdef double PoissonMean(double lamb):
    return lamb

cpdef double PoissonVar(double lamb):
    return lamb

cpdef double PoissonStd(double lamb):
    return sqrt(lamb)

cpdef int PoissonMode(double lamb):
    # TODO: Decide if this is reasonable
    return <int>(lamb-.5)

# ===== Exponential Distribution =====

cpdef double ExponentialRand(double lamb, RandomEngine engine=defaultEngine):
    engine.exponential_rand.param(expon_rng.param_type(lamb))
    return engine.exponential_rand(engine.source)

cpdef double ExponentialPDF(double x, double lamb):
    if x > 0:
        return lamb * exp(-lamb*x)
    return 0.

cpdef double ExponentialLogPDF(double x, double lamb):
    if x > 0:
        return log(lamb)-lamb*x
    return 0.

cpdef double ExponentialCDF(double x, double lamb):
    if x > 0:
        return 1.-exp(-lamb*x)
    return 0

cpdef double ExponentialDLDL(double x, double lamb):
    return 1./lamb - x

cpdef double ExponentialMean(double lamb):
    return 1./lamb

cpdef double ExponentialVar(double lamb):
    return 1./(lamb*lamb)

cpdef double ExponentialStd(double lamb):
    return 1./lamb

cpdef double ExponentialMode(double lamb):
    return 0

# ===== Binomial Distribution =====

cpdef double BinomialRand(int n, double probability, RandomEngine engine=defaultEngine):
    engine.binomial_rand.param(binom_rng.param_type(n,probability))
    return engine.binomial_rand(engine.source)

cpdef double BinomialPDF(int x, int n, double probability):
    cdef binom_info* temp = new binom_info(n,probability)
    cdef double return_value = pdf(temp[0],x)
    del temp
    return return_value

cpdef double BinomialLogPDF(int x, int n, double probability):
    return log(BinomialPDF(x,n,probability))

cpdef double BinomialCDF(double x, int n, double probability):
    cdef binom_info* temp = new binom_info(n,probability)
    cdef double return_value = cdf(temp[0],<int>x)
    del temp
    return return_value

cpdef double BinomialDLDP(int x, int n, double probability):
    return x/probability - (1-x)/(1.-probability)

cpdef double BinomialMean(int n, double probability):
    return n*probability

cpdef double BinomialVar(int n, double probability):
    return n*probability*(1.-probability)

cpdef double BinomialStd(int n, double probability):
    return sqrt(n*probability*(1.-probability))

cpdef double BinomialMode(int n, double probability):
    cdef int x = <int>(n*probability)
    if BinomialPDF(x,n,probability) >= BinomialPDF(x,n,probability):
        return x
    else:
        return x+1
