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
        self._rand.param(gamma_rng.param_type(shape,rate))
        return self._rand(self._generator)

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
