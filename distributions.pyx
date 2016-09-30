cdef class _uniform:
    cpdef double pdf(self,double x, double lower=0, double upper=1):
        if x > lower and x < upper:
            return 1./(upper-lower)
        return 0.

    cpdef double logPDF(self,double x, double lower=0, double upper=1):
        return log(self.pdf(x,lower,upper))

    cpdef double cdf(self,double x, double lower=0, double upper=1):
        if x < lower:
            return 0.
        if x > upper:
            return 1.
        return (x-lower)/(upper-lower)

    cpdef double dldu(self,double x, double lower=0, double upper=1):
        return 0.

    cpdef double dldl(self,double x, double lower=0, double upper=1):
        return 0.

    cpdef double dldx(self,double x, double lower=0, double upper=1):
        return 0.

    cpdef double rand(self,double lower=0, double upper=1):
        return self._rand(self._generator)*(upper-lower) + lower

    cpdef double mean(self, double lower=0, double upper=1):
        return (upper+lower)/2.

    cpdef double var(self, double lower=0, double upper=1):
        return (upper-lower)**2 / 12.

    cpdef double std(self, double lower=0, double upper=1):
        return (upper-lower)/sqrt(12.)

    cpdef double mode(self, double lower=0, double upper=1):
        # TODO: return nan?
        return (upper+lower)/2.

    def __cinit__(self,unsigned long int seed):
        self._generator = mTwister(seed)
        return


cdef class _normal:
    cpdef double pdf(self,double x, double mean=1, double sigma=1):
        return exp(-(x-mean)**2/(2*sigma*sigma))/sqrt(2*pi*sigma*sigma)

    cpdef double logPDF(self,double x, double mean=1, double sigma=1):
        return log(self.pdf(x,mean,sigma))

    cpdef double cdf(self,double x, double mean=1, double sigma=1):
        return cdf(normal(mean,sigma),x)

    cpdef double dldm(self,double x, double mean=1, double sigma=1):
        return (x-mean)/(sigma*sigma)

    cpdef double dldv(self,double x, double mean=1, double sigma=1):
        return (x-mean)**2 / (2*sigma**4) - .5/sigma**2

    cpdef double dlds(self,double x, double mean=1, double sigma=1):
        return (x-mean)**2 / sigma**3 - 1./sigma

    cpdef double rand(self,double mean=1, double sigma=1):
        return self._rand(self._generator)*sigma+mean

    cpdef double mean(self, double mean=1, double sigma=1):
        return mean

    cpdef double var(self, double mean=1, double sigma=1):
        return sigma*sigma

    cpdef double std(self, double mean=1, double sigma=1):
        return sigma

    cpdef double mode(self, double mean=1, double sigma=1):
        return mean

    def __cinit__(self,unsigned long int seed):
        self._generator = mTwister(seed)
        return


cdef class _gamma:
    cpdef double pdf(self,double x, double shape, double rate):
        cdef gamma_info* temp = new gamma_info(shape,1./rate)
        cdef double return_value = pdf(temp[0],x)
        del temp
        return return_value

    cpdef double logPDF(self,double x, double shape, double rate):
        return log(self.pdf(x,shape,rate))

    cpdef double cdf(self,double x, double shape, double rate):
        cdef gamma_info* temp = new gamma_info(shape,1./rate)
        cdef double return_value = cdf(temp[0],x)
        del temp
        return return_value

    cpdef double dlda(self,double x, double shape, double rate):
        return log(rate) + log(x) - digamma(shape)

    cpdef double dldb(self,double x, double shape, double rate):
        return shape/rate - x

    cpdef double dldx(self,double x, double shape, double rate):
        return (shape-1)/x - rate

    cpdef double rand(self,double shape, double rate):
        self._rand.param(gamma_rng.param_type(shape,1./rate))
        return self._rand(self._generator)

    cpdef double mean(self, double shape, double rate):
        return shape/rate

    cpdef double var(self, double shape, double rate):
        return shape/rate**2

    cpdef double std(self, double shape, double rate):
        return sqrt(shape)/rate

    cpdef double mode(self, double shape, double rate):
        if shape < 1:
            return 0
        return (shape-1)/rate

    def __cinit__(self,unsigned long int seed):
        self._generator = mTwister(seed)


cdef class _invGamma:
    cpdef double pdf(self,double x, double shape, double rate):
        cdef gamma_info* temp = new gamma_info(shape,rate)
        cdef double return_value = pdf(temp[0],1./x)
        del temp
        return return_value

    cpdef double logPDF(self,double x, double shape, double rate):
        return log(self.pdf(x,shape,rate))

    cpdef double cdf(self,double x, double shape, double rate):
        cdef gamma_info* temp = new gamma_info(shape,rate)
        cdef double return_value = cdf(temp[0],1./x)
        del temp
        return return_value

    cpdef double dlda(self,double x, double shape, double rate):
        return log(rate) - log(x) - digamma(shape)

    cpdef double dldb(self,double x, double shape, double rate):
        return shape/rate - 1./x

    cpdef double dldx(self,double x, double shape, double rate):
        return 2.*rate/x - (shape + 1.)/x

    cpdef double rand(self,double shape, double rate):
        self._rand.param(gamma_rng.param_type(shape,1./rate))
        return 1.0/self._rand(self._generator)

    cpdef double mean(self, double shape, double rate):
        return rate / (shape - 1)

    cpdef double var(self, double shape, double rate):
        return rate**2  / ((shape-1)**2 * (shape-2))

    cpdef double std(self, double shape, double rate):
        if shape <= 2:
            # TODO: make nan
            return 0
        return rate / ((shape-1)*sqrt(shape-2))

    cpdef double mode(self, double shape, double rate):
        return rate/(shape+1.)

    def __init__(self,unsigned long int seed):
        self._generator = mTwister(seed)


cdef class _beta:
    cpdef double pdf(self,double x, double alpha, double beta):
        cdef beta_info* temp = new beta_info(alpha,beta)
        cdef double return_value = pdf(temp[0],x)
        del temp
        return return_value

    cpdef double logPDF(self,double x, double alpha, double beta):
        return log(self.pdf(x,alpha, beta))

    cpdef double cdf(self,double x, double alpha, double beta):
        cdef beta_info* temp = new beta_info(alpha,beta)
        cdef double return_value = cdf(temp[0],x)
        del temp
        return return_value

    cpdef double dlda(self,double x, double alpha, double beta):
        return log(x) + digamma(alpha+beta) - digamma(alpha)

    cpdef double dldb(self,double x, double alpha, double beta):
        return log(1-x) + digamma(alpha+beta) - digamma(alpha)

    cpdef double rand(self,double alpha, double beta):
        self._rand.param(beta_rng.param_type(alpha,beta))
        return self._rand(self._generator)

    cpdef double mean(self, double alpha, double beta):
        return alpha / (alpha+beta)

    cpdef double var(self, double alpha, double beta):
        return alpha*beta / ((alpha+beta)**2*(alpha+beta+1))

    cpdef double std(self, double alpha, double beta):
        return sqrt(self.var(alpha,beta))

    cpdef double mode(self, double alpha, double beta):
        if alpha <= 1 or beta <= 1.:
            if alpha < beta:
                return 0
            elif alpha > beta:
                return 1
            else:
                return .5
        return (alpha - 1) / (alpha + beta - 2)

    def __cinit__(self,unsigned long int seed):
        self._generator = mTwister(seed)


cdef class _poisson:
    cpdef double pdf(self, int x, double lamb):
        cdef pois_info* temp = new pois_info(lamb)
        cdef double return_value = pdf(temp[0],x)
        del temp
        return return_value

    cpdef double logPDF(self, int x, double lamb):
        return log(self.pdf(x,lamb))

    cpdef double cdf(self, double x, double lamb):
        cdef pois_info* temp = new pois_info(lamb)
        cdef double return_value = cdf(temp[0],<int>x)
        del temp
        return return_value

    cpdef double dldl(self, int x, double lamb):
        return x/lamb - 1.

    cpdef double rand(self,double lamb):
        self._rand.param(pois_rng.param_type(lamb))
        return self._rand(self._generator)

    cpdef double mean(self, double lamb):
        return lamb

    cpdef double var(self, double lamb):
        return lamb

    cpdef double std(self, double lamb):
        return sqrt(lamb)

    cpdef double mode(self, double lamb):
        # TODO: Decide if this is reasonable
        return lamb-.5

    def __cinit__(self,unsigned long int seed):
        self._generator = mTwister(seed)


cdef class _expon:
    cpdef double pdf(self, double x, double lamb):
        if x > 0:
            return lamb * exp(-lamb*x)
        return 0.

    cpdef double logPDF(self, double x, double lamb):
        if x > 0:
            return log(lamb)-lamb*x
        return 0.

    cpdef double cdf(self, double x, double lamb):
        if x > 0:
            return 1.-exp(-lamb*x)
        return 0

    cpdef double dldl(self, double x, double lamb):
        return 1./lamb - x

    cpdef double rand(self,double lamb):
        self._rand.param(expon_rng.param_type(lamb))
        return self._rand(self._generator)

    cpdef double mean(self, double lamb):
        return 1./lamb

    cpdef double var(self, double lamb):
        return 1./(lamb*lamb)

    cpdef double std(self, double lamb):
        return 1./lamb

    cpdef double mode(self, double lamb):
        return 0

    def __cinit__(self,unsigned long int seed):
        self._generator = mTwister(seed)


cdef class _binomial:
    cpdef double pdf(self, int x, int n, double probability):
        cdef binom_info* temp = new binom_info(n,probability)
        cdef double return_value = pdf(temp[0],x)
        del temp
        return return_value

    cpdef double logPDF(self, int x, int n, double probability):
        return log(self.pdf(x,n,probability))

    cpdef double cdf(self, double x, int n, double probability):
        cdef binom_info* temp = new binom_info(n,probability)
        cdef double return_value = cdf(temp[0],<int>x)
        del temp
        return return_value

    cpdef double dldp(self, int x, int n, double probability):
        return x/probability - (1-x)/(1.-probability)

    cpdef double rand(self,int n, double probability):
        self._rand.param(binom_rng.param_type(n,probability))
        return self._rand(self._generator)

    cpdef double mean(self, int n, double probability):
        return n*probability

    cpdef double var(self, int n, double probability):
        return n*probability*(1.-probability)

    cpdef double std(self, int n, double probability):
        return sqrt(n*probability*(1.-probability))

    cpdef double mode(self, int n, double probability):
        cdef int x = <int>(n*probability)
        if self.pdf(x,n,probability) >= self.pdf(x,n,probability):
            return x
        else:
            return x+1

    def __cinit__(self,unsigned long int seed):
        self._generator = mTwister(seed)


