
cdef class _normal:
    cpdef double pdf(self,double x, double mean, double sigma):
        return gsl_ran_gaussian_pdf(x-mean,sigma)

    cpdef double logPDF(self,double x, double mean, double sigma):
        return log(gsl_ran_gaussian_pdf(x-mean,sigma))

    cpdef double dldm(self,double x, double mean, double sigma):
        return (x-mean)/(sigma*sigma)

    cpdef double dldv(self,double x, double mean, double sigma):
        return -sigma*sigma/2.0 + (x-mean)/sigma**4.0

    cpdef double rand(self,double mean, double sigma):
        return gsl_ran_gaussian(self._RNG,sigma)+mean

    def __init__(self,unsigned long int seed):
        self._RNG = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(self._RNG,seed)


cdef class _gamma:
    cpdef double pdf(self,double x, double shape, double rate):
        return gsl_ran_gamma_pdf(x,shape, 1./rate)

    cpdef double logPDF(self,double x, double shape, double rate):
        return log(gsl_ran_gamma_pdf(x,shape, 1./rate))

    # cpdef double dlda(self,double x, double shape, double rate)

    # cpdef double dldb(self,double x, double shape, double rate)

    cpdef double rand(self,double shape, double rate):
        return gsl_ran_gamma(self._RNG, shape, 1./rate)

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

    # cpdef double dlda(self,double x, double alpha, double beta)

    # cpdef double dldb(self,double x, double alpha, double beta)

    cpdef double rand(self,double alpha, double beta):
        return gsl_ran_beta(self._RNG, alpha, beta)

    def __init__(self,unsigned long int seed):
        self._RNG = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(self._RNG,seed)
