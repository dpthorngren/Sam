cdef extern from "gsl/gsl_rng.h":
    struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    gsl_rng_type* gsl_rng_mt19937
    gsl_rng* gsl_rng_alloc(gsl_rng_type* rngType)
    void gsl_rng_set(gsl_rng* rng, unsigned long int s)

cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_gaussian_pdf(double x, double sigma)
    double gsl_cdf_gaussian_P(double x, double sigma)
    double gsl_ran_gaussian(gsl_rng* rng, double sigma)
    double gsl_ran_gamma(gsl_rng* rng, double a, double b)
    double gsl_ran_gamma_pdf(double x, double a, double b)
    double gsl_cdf_gamma_P(double x, double a, double b)
    double gsl_ran_beta(gsl_rng* rng, double a, double b)
    double gsl_ran_beta_pdf(double x, double a, double b)

cdef extern from "gsl/gsl_sf.h":
    double gsl_sf_psi(double x)
    double gsl_sf_beta_inc(double a, double b, double x)

cdef extern from "math.h":
    double log(double x)
    double log10(double x)
    double sqrt(double x)
    double exp(double x)
    double sin(double x)
    double cos(double x)
    double tan(double x)

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
    cdef gsl_rng* _RNG
    cpdef double pdf(self,double x, double mean, double sigma)
    cpdef double logPDF(self,double x, double mean, double sigma)
    cpdef double cdf(self,double x, double mean, double sigma)
    cpdef double dldm(self,double x, double mean, double sigma)
    cpdef double dldv(self,double x, double mean, double sigma)
    cpdef double dlds(self,double x, double mean, double sigma)
    cpdef double rand(self,double mean, double sigma)
    cpdef double mean(self, double mean, double sigma)
    cpdef double var(self, double mean, double sigma)
    cpdef double std(self, double mean, double sigma)
    cpdef double mode(self, double mean, double sigma)


cdef class _invChiSq:
    cdef gsl_rng* _RNG
    cpdef double pdf(self,double x, double nu, double tau)
    cpdef double logPDF(self,double x, double nu, double tau)
    # cpdef double dlda(self,double x, double nu, double tau)
    # cpdef double dldb(self,double x, double nu, double tau)
    cpdef double rand(self,double nu, double tau)


cdef class _invGamma:
    cdef gsl_rng* _RNG
    cpdef double pdf(self,double x, double shape, double rate)
    cpdef double logPDF(self,double x, double shape, double rate)
    # cpdef double dlda(self,double x, double shape, double rate)
    # cpdef double dldb(self,double x, double shape, double rate)
    cpdef double rand(self,double shape, double rate)
    cpdef double mean(self, double shape, double rate)
    cpdef double var(self, double shape, double rate)
    cpdef double std(self, double shape, double rate)
    cpdef double mode(self, double shape, double rate)


cdef class _gamma:
    cdef gsl_rng* _RNG
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
    cdef gsl_rng* _RNG
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
