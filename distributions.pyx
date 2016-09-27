
cdef class _normal:
    cpdef double pdf(self,double x, double mean, double sigma):
        return gsl_ran_gaussian_pdf(x-mean,sigma)

    cpdef double logPDF(self,double x, double mean, double sigma):
        return log(gsl_ran_gaussian_pdf(x-mean,sigma))

    cpdef double cdf(self,double x, double mean, double sigma):
        return gsl_cdf_gaussian_P(x-mean,sigma)

    cpdef double dldm(self,double x, double mean, double sigma):
        return (x-mean)/(sigma*sigma)

    cpdef double dldv(self,double x, double mean, double sigma):
        return (x-mean)**2 / (2*sigma**4) - .5/sigma**2

    cpdef double dlds(self,double x, double mean, double sigma):
        return (x-mean)**2 / sigma**3 - 1./sigma

    cpdef double rand(self,double mean, double sigma):
        return gsl_ran_gaussian(self._RNG,sigma)+mean

    cpdef double mean(self, double mean, double sigma):
        return mean

    cpdef double var(self, double mean, double sigma):
        return sigma*sigma

    cpdef double std(self, double mean, double sigma):
        return sigma

    cpdef double mode(self, double mean, double sigma):
        return mean

    def __init__(self,unsigned long int seed):
        self._RNG = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(self._RNG,seed)


cdef class _gamma:
    cpdef double pdf(self,double x, double shape, double rate):
        return gsl_ran_gamma_pdf(x,shape, 1./rate)

    cpdef double logPDF(self,double x, double shape, double rate):
        return log(gsl_ran_gamma_pdf(x,shape, 1./rate))

    cpdef double cdf(self,double x, double shape, double rate):
        return gsl_cdf_gamma_P(x,shape,1./rate)

    cpdef double dlda(self,double x, double shape, double rate):
        return log(rate) + log(x) - gsl_sf_psi(shape)

    cpdef double dldb(self,double x, double shape, double rate):
        return shape/rate - x

    cpdef double dldx(self,double x, double shape, double rate):
        return (shape-1)/x - rate

    cpdef double rand(self,double shape, double rate):
        return gsl_ran_gamma(self._RNG, shape, 1./rate)

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

    def __init__(self,unsigned long int seed):
        self._RNG = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(self._RNG,seed)


cdef class _invGamma:
    cpdef double pdf(self,double x, double shape, double rate):
        return gsl_ran_gamma_pdf(1./x,shape, 1./rate)

    cpdef double logPDF(self,double x, double shape, double rate):
        return log(gsl_ran_gamma_pdf(1./x,shape, 1./rate))

    # cpdef double dlda(self,double x, double shape, double rate)

    # cpdef double dldb(self,double x, double shape, double rate)

    cpdef double rand(self,double shape, double rate):
        return 1./gsl_ran_gamma(self._RNG, shape, 1./rate)

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
        self._RNG = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(self._RNG,seed)


cdef class _invChiSq:
    cpdef double pdf(self,double x, double nu, double tau):
        return gsl_ran_gamma_pdf(1./x,nu/2.0, 2./(nu*tau*tau))

    cpdef double logPDF(self,double x, double nu, double tau):
        return log(gsl_ran_gamma_pdf(1./x,nu/2.0, 2./(nu*tau*tau)))

    # cpdef double dlda(self,double x, double nu, double tau)

    # cpdef double dldb(self,double x, double nu, double tau)

    cpdef double rand(self,double nu, double tau):
        return 1./gsl_ran_gamma(self._RNG, nu/2.0, 1./tau)

    def __init__(self,unsigned long int seed):
        self._RNG = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(self._RNG,seed)


cdef class _beta:
    cpdef double pdf(self,double x, double alpha, double beta):
        return gsl_ran_beta_pdf(x,alpha, beta)

    cpdef double logPDF(self,double x, double alpha, double beta):
        return log(gsl_ran_beta_pdf(x,alpha, beta))

    cpdef double cdf(self,double x, double alpha, double beta):
        return gsl_sf_beta_inc(alpha,beta,x)

    cpdef double dlda(self,double x, double alpha, double beta):
        return log(x) + gsl_sf_psi(alpha+beta) - gsl_sf_psi(alpha)

    cpdef double dldb(self,double x, double alpha, double beta):
        return log(1-x) + gsl_sf_psi(alpha+beta) - gsl_sf_psi(alpha)

    cpdef double rand(self,double alpha, double beta):
        return gsl_ran_beta(self._RNG, alpha, beta)

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

    def __init__(self,unsigned long int seed):
        self._RNG = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(self._RNG,seed)
