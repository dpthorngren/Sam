cdef class RandomEngine:
    def __init__(self, unsigned long int i):
        self.setSeed(i)

    cpdef object setSeed(self,unsigned long int i):
        self.source = mTwister(i)

if (sys.version_info > (3, 0)):
    defaultEngine = RandomEngine(<unsigned long int>int(os.urandom(4).hex(),16))
else:
    defaultEngine = RandomEngine(<unsigned long int>int(os.urandom(4).encode("hex"),16))

# ===== Uniform Distribution =====

cpdef double uniformRand(double lower=0, double upper=1, RandomEngine engine=defaultEngine) except? 999.:
    return engine.uniform_rand(engine.source)*(upper-lower) + lower

cpdef double uniformPDF(double x, double lower=0, double upper=1) except? 999.:
        if x > lower and x < upper:
            return 1./(upper-lower)
        return 0.

cpdef double uniformLogPDF(double x, double lower=0, double upper=1) except? 999.:
    return log(uniformPDF(x,lower,upper))

cpdef double uniformCDF(double x, double lower=0, double upper=1) except? 999.:
    if x < lower:
        return 0.
    if x > upper:
        return 1.
    return (x-lower)/(upper-lower)

cpdef double uniformDLDU(double x, double lower=0, double upper=1) except? 999.:
    if x < lower or x > upper:
        return 0
    return 1./(lower - upper)

cpdef double uniformDLDL(double x, double lower=0, double upper=1) except? 999.:
    if x < lower or x > upper:
        return 0
    return 1./(upper - lower)

cpdef double uniformMean(double lower=0, double upper=1) except? 999.:
    return (upper+lower)/2.

cpdef double uniformVar(double lower=0, double upper=1) except? 999.:
    return (upper-lower)**2 / 12.

cpdef double uniformStd(double lower=0, double upper=1) except? 999.:
    return (upper-lower)/sqrt(12.)

# ===== Normal Distribution =====

cpdef double normalRand(double mean=0, double sigma=1, RandomEngine engine = defaultEngine) except? 999.:
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")
    return engine.normal_rand(engine.source)*sigma + mean

cpdef double normalPDF(double x, double mean=0, double sigma=1) except? 999.:
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")
    return exp(-(x-mean)**2/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma)

cpdef double normalLogPDF(double x, double mean=0, double sigma=1) except? 999.:
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")
    return -(x-mean)**2/(2*sigma*sigma) - .5*log(2*pi*sigma*sigma)

cpdef double normalCDF(double x, double mean=0, double sigma=1) except? 999.:
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")
    return cdf(normal_info(mean,sigma),x)

cpdef double normalDLDM(double x, double mean=0, double sigma=1) except? 999.:
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")
    return (x-mean)/(sigma*sigma)

cpdef double normalDLDX(double x, double mean=0, double sigma=1) except? 999.:
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")
    return (mean-x)/(sigma*sigma)

cpdef double normalDLDV(double x, double mean=0, double sigma=1) except? 999.:
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")
    return (x-mean)**2 / (2*sigma**4) - .5/sigma**2

cpdef double normalDLDS(double x, double mean=0, double sigma=1) except? 999.:
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")
    return (x-mean)**2 / sigma**3 - 1./sigma

cpdef double normalMean(double mean=0, double sigma=1) except? 999.:
    return mean

cpdef double normalVar(double mean=0, double sigma=1) except? 999.:
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")
    return sigma*sigma

cpdef double normalStd(double mean=0, double sigma=1) except? 999.:
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")
    return sigma

cpdef double normalMode(double mean=0, double sigma=1) except? 999.:
    return mean

# ===== Multivariate Normal Distribution =====

cpdef double[:] mvNormalRand(double[:] mean, double[:,:] covariance, double[:] output = None, bint isChol=False, RandomEngine engine=defaultEngine) except *:
    if (mean is None) or (covariance is None):
        raise ValueError("Only the output may be None.")
    cdef Size i, j
    cdef Size d = mean.shape[0]
    cdef double[:,:] covChol
    if (d != covariance.shape[0]) or (d != covariance.shape[1]):
        raise ValueError("Covariance matrix shape does not max mean vector.")
    if output is None:
        output = mean.copy()
    elif d != output.shape[0]:
        raise ValueError("Output vector length does not max mean vector.")
    if isChol:
        covChol = covariance
    else:
        covChol = covariance.copy()
        choleskyInplace(covChol)
    for i in range(d):
        output[i] = normalRand(engine=engine)
    cdef int n = d
    cdef int increment = 1
    blas.dtrmv('U','T','N',&n,&covChol[0,0],&n,&output[0],&increment)
    for i in range(d):
        output[i] += mean[i]
    return output

cpdef double mvNormalPDF(double[:] x, double[:] mean, double[:,:] covariance, bint isChol=False) except? 999.:
    return exp(mvNormalLogPDF(x,mean,covariance,isChol))

cpdef double mvNormalLogPDF(double[:] x, double[:] mean, double[:,:] covariance, bint isChol=False) except? 999.:
    cdef Size i;
    cdef Size d = mean.shape[0]
    if (x is None) or (mean is None) or (covariance is None):
        raise ValueError("Inputs may not be None!")
    if (d != covariance.shape[0]) or (d != covariance.shape[1]):
        raise ValueError("Covariance matrix shape does not max mean vector.")
    if d != x.shape[0]:
        raise ValueError("X vector length does not max mean vector.")
    cdef double[:,:] covChol
    if isChol:
        covChol = covariance
    else:
        covChol = covariance.copy()
        choleskyInplace(covChol)
    cdef double result = 0.
    cdef double[:] offset = x.copy()
    for i in range(d):
        # Compute determinant portion of the pdf:
        result -= log(covChol[i,i])
        offset[i] -= mean[i]

    # (x-mean) * L.T^-1 * L^-1 * (x-mean)
    cdef int n = d
    cdef int increment = 1
    blas.dtrsv('U','T','N',&n,&covChol[0,0],&n,&offset[0],&increment)
    blas.dtrsv('U','N','N',&n,&covChol[0,0],&n,&offset[0],&increment)
    for i in range(d):
        result -= .5*offset[i]*(x[i] - mean[i])
    return result -.5*d*log(2.*pi)

# ===== Gamma Distribution =====

cpdef double gammaRand(double shape, double rate, RandomEngine engine = defaultEngine) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    engine.gamma_rand.param(gamma_rng.param_type(shape,1./rate))
    return engine.gamma_rand(engine.source)

cpdef double gammaPDF(double x, double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    cdef gamma_info* temp = new gamma_info(shape,1./rate)
    cdef double return_value = pdf(temp[0],x)
    del temp
    return return_value

cpdef double gammaLogPDF(double x, double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    return log(gammaPDF(x,shape,rate))

cpdef double gammaCDF(double x, double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    cdef gamma_info* temp = new gamma_info(shape,1./rate)
    cdef double return_value = cdf(temp[0],x)
    del temp
    return return_value

cpdef double gammaDLDA(double x, double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    return log(rate) + log(x) - digamma(shape)

cpdef double gammaDLDB(double x, double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    return shape/rate - x

cpdef double gammaDLDX(double x, double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    return (shape-1)/x - rate

cpdef double gammaMean(double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    return shape/rate

cpdef double gammaVar(double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    return shape/rate**2

cpdef double gammaStd(double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    return sqrt(shape)/rate

cpdef double gammaMode(double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    if shape < 1:
        return 0
    return (shape-1)/rate

# ===== Inverse-Gamma Distribution =====

cpdef double invGammaRand(double shape, double rate, RandomEngine engine=defaultEngine) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    engine.gamma_rand.param(gamma_rng.param_type(shape,1./rate))
    return 1.0/engine.gamma_rand(engine.source)

cpdef double invGammaPDF(double x, double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    cdef gamma_info* temp = new gamma_info(shape,rate)
    cdef double return_value = pdf(temp[0],1./x)
    del temp
    return return_value

cpdef double invGammaLogPDF(double x, double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    return log(invGammaPDF(x,shape,rate))

cpdef double invGammaCDF(double x, double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    cdef gamma_info* temp = new gamma_info(shape,rate)
    cdef double return_value = cdf(temp[0],1./x)
    del temp
    return return_value

cpdef double invGammaDLDA(double x, double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    return log(rate) - log(x) - digamma(shape)

cpdef double invGammaDLDB(double x, double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    return shape/rate - 1./x

cpdef double invGammaDLDX(double x, double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    return 2.*rate/x - (shape + 1.)/x

cpdef double invGammaMean(double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    return rate / (shape - 1)

cpdef double invGammaVar(double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    return rate**2  / ((shape-1)**2 * (shape-2))

cpdef double invGammaStd(double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    if shape <= 2:
        return nan
    return rate / ((shape-1)*sqrt(shape-2))

cpdef double invGammaMode(double shape, double rate) except? 999.:
    if (shape <= 0):
        raise ValueError("Shape must be > 0.")
    if (rate <= 0):
        raise ValueError("Rate must be > 0.")
    return rate/(shape+1.)

# ===== Beta Distribution =====

cpdef double betaRand(double alpha, double beta, RandomEngine engine = defaultEngine) except? 999.:
    # Boost handles the bounds checking
    engine.beta_rand.param(beta_rng.param_type(alpha,beta))
    return engine.beta_rand(engine.source)

cpdef double betaPDF(double x, double alpha, double beta) except? 999.:
    # Boost handles the bounds checking
    cdef beta_info* temp = new beta_info(alpha,beta)
    cdef double return_value = pdf(temp[0],x)
    del temp
    return return_value

cpdef double betaLogPDF(double x, double alpha, double beta) except? 999.:
    if (alpha <= 0):
        raise ValueError("Alpha must be > 0.")
    if (beta <= 0):
        raise ValueError("Beta must be > 0.")
    return log(betaPDF(x,alpha, beta))

cpdef double betaCDF(double x, double alpha, double beta) except? 999.:
    # Boost handles the bounds checking
    cdef beta_info* temp = new beta_info(alpha,beta)
    cdef double return_value = cdf(temp[0],x)
    del temp
    return return_value

cpdef double betaDLDA(double x, double alpha, double beta) except? 999.:
    if (alpha <= 0):
        raise ValueError("Alpha must be > 0.")
    if (beta <= 0):
        raise ValueError("Beta must be > 0.")
    return log(x) + digamma(alpha+beta) - digamma(alpha)

cpdef double betaDLDB(double x, double alpha, double beta) except? 999.:
    if (alpha <= 0):
        raise ValueError("Alpha must be > 0.")
    if (beta <= 0):
        raise ValueError("Beta must be > 0.")
    return log(1-x) + digamma(alpha+beta) - digamma(alpha)

cpdef double betaMean(double alpha, double beta) except? 999.:
    if (alpha <= 0):
        raise ValueError("Alpha must be > 0.")
    if (beta <= 0):
        raise ValueError("Beta must be > 0.")
    return alpha / (alpha+beta)

cpdef double betaVar(double alpha, double beta) except? 999.:
    if (alpha <= 0):
        raise ValueError("Alpha must be > 0.")
    if (beta <= 0):
        raise ValueError("Beta must be > 0.")
    return alpha*beta / ((alpha+beta)**2*(alpha+beta+1))

cpdef double betaStd(double alpha, double beta) except? 999.:
    if (alpha <= 0):
        raise ValueError("Alpha must be > 0.")
    if (beta <= 0):
        raise ValueError("Beta must be > 0.")
    return sqrt(betaVar(alpha,beta))

cpdef double betaMode(double alpha, double beta) except? 999.:
    if (alpha <= 0):
        raise ValueError("Alpha must be > 0.")
    if (beta <= 0):
        raise ValueError("Beta must be > 0.")
    if alpha <= 1 or beta <= 1.:
        if alpha < beta:
            return 0
        elif alpha > beta:
            return 1
        else:
            return .5
    return (alpha - 1) / (alpha + beta - 2)

# ===== Poisson Distribution =====

cpdef double poissonRand(double lamb, RandomEngine engine=defaultEngine) except? 999.:
    # Boost handles the bounds checking
    engine.poisson_rand.param(pois_rng.param_type(lamb))
    return engine.poisson_rand(engine.source)

cpdef double poissonPDF(int x, double lamb) except? 999.:
    # Boost handles the bounds checking
    cdef pois_info* temp = new pois_info(lamb)
    cdef double return_value = pdf(temp[0],x)
    del temp
    return return_value

cpdef double poissonLogPDF(int x, double lamb) except? 999.:
    # Boost handles the bounds checking
    return log(poissonPDF(x,lamb))

cpdef double poissonCDF(double x, double lamb) except? 999.:
    # Boost handles the bounds checking
    cdef pois_info* temp = new pois_info(lamb)
    cdef double return_value = cdf(temp[0],<int>x)
    del temp
    return return_value

cpdef double poissonDLDL(int x, double lamb) except? 999.:
    if lamb <= 0:
        raise ValueError("Lambda must be > 0.")
    if x < 0:
        raise ValueError("x must be > 0.")
    return x/lamb - 1.

cpdef double poissonMean(double lamb) except? 999.:
    if lamb <= 0:
        raise ValueError("Lambda must be > 0.")
    return lamb

cpdef double poissonVar(double lamb) except? 999.:
    if lamb <= 0:
        raise ValueError("Lambda must be > 0.")
    return lamb

cpdef double poissonStd(double lamb) except? 999.:
    if lamb <= 0:
        raise ValueError("Lambda must be > 0.")
    return sqrt(lamb)

cpdef int poissonMode(double lamb) except? -1:
    '''Returns max mode if result is not unique (integer lambda).'''
    if lamb <= 0:
        raise ValueError("Lambda must be > 0.")
    return <int>(lamb)

# ===== Exponential Distribution =====

cpdef double exponentialRand(double lamb, RandomEngine engine=defaultEngine) except? 999.:
    # Boost handles the bounds checking
    engine.exponential_rand.param(expon_rng.param_type(lamb))
    return engine.exponential_rand(engine.source)

cpdef double exponentialPDF(double x, double lamb) except? 999.:
    if lamb <= 0:
        raise ValueError("Lambda must be > 0.")
    if x >= 0:
        return lamb * exp(-lamb*x)
    return 0.

cpdef double exponentialLogPDF(double x, double lamb) except? 999.:
    if lamb <= 0:
        raise ValueError("Lambda must be > 0.")
    if x >= 0:
        return log(lamb)-lamb*x
    return 0.

cpdef double exponentialCDF(double x, double lamb) except? 999.:
    if lamb <= 0:
        raise ValueError("Lambda must be > 0.")
    if x >= 0:
        return 1.-exp(-lamb*x)
    return 0.

cpdef double exponentialDLDL(double x, double lamb) except? 999.:
    if lamb <= 0:
        raise ValueError("Lambda must be > 0.")
    if x >= 0:
        return 1./lamb - x
    return 0.

cpdef double exponentialMean(double lamb) except? 999.:
    if lamb <= 0:
        raise ValueError("Lambda must be > 0.")
    return 1./lamb

cpdef double exponentialVar(double lamb) except? 999.:
    if lamb <= 0:
        raise ValueError("Lambda must be > 0.")
    return 1./(lamb*lamb)

cpdef double exponentialStd(double lamb) except? 999.:
    if lamb <= 0:
        raise ValueError("Lambda must be > 0.")
    return 1./lamb

cpdef double exponentialMode(double lamb) except? 999.:
    if lamb <= 0:
        raise ValueError("Lambda must be > 0.")
    return 0

# ===== Binomial Distribution =====

cpdef int binomialRand(int n, double probability, RandomEngine engine=defaultEngine) except? -1:
    # Boost handles the bounds checking
    engine.binomial_rand.param(binom_rng.param_type(n,probability))
    return engine.binomial_rand(engine.source)

cpdef double binomialPDF(int x, int n, double probability) except? 999.:
    # Boost handles the bounds checking
    cdef binom_info* temp = new binom_info(n,probability)
    cdef double return_value = pdf(temp[0],x)
    del temp
    return return_value

cpdef double binomialLogPDF(int x, int n, double probability) except? 999.:
    # Boost handles the bounds checking
    return log(binomialPDF(x,n,probability))

cpdef double binomialCDF(double x, int n, double probability) except? 999.:
    # Boost handles the bounds checking
    cdef binom_info* temp = new binom_info(n,probability)
    cdef double return_value = cdf(temp[0],<int>x)
    del temp
    return return_value

cpdef double binomialDLDP(int x, int n, double probability) except? 999.:
    if (probability < 0) or (probability > 1):
        raise ValueError("Probaiblity must be between 0 and 1.")
    if n <= 0:
        raise ValueError("n must be > 0.")
    if (x >= 0) and (x <= n):
        return x/probability - (1-x)/(1.-probability)
    return 0.

cpdef double binomialMean(int n, double probability) except? 999.:
    if (probability < 0) or (probability > 1):
        raise ValueError("Probaiblity must be between 0 and 1.")
    if n <= 0:
        raise ValueError("n must be > 0.")
    return n*probability

cpdef double binomialVar(int n, double probability) except? 999.:
    if (probability < 0) or (probability > 1):
        raise ValueError("Probaiblity must be between 0 and 1.")
    if n <= 0:
        raise ValueError("n must be > 0.")
    return n*probability*(1.-probability)

cpdef double binomialStd(int n, double probability) except? 999.:
    if (probability < 0) or (probability > 1):
        raise ValueError("Probaiblity must be between 0 and 1.")
    if n <= 0:
        raise ValueError("n must be > 0.")
    return sqrt(n*probability*(1.-probability))

cpdef double binomialMode(int n, double probability) except? 999.:
    if (probability < 0) or (probability > 1):
        raise ValueError("Probaiblity must be between 0 and 1.")
    if n <= 0:
        raise ValueError("n must be > 0.")
    cdef int x = <int>(n*probability)
    if binomialPDF(x,n,probability) >= binomialPDF(x+1,n,probability):
        return x
    else:
        return x+1
