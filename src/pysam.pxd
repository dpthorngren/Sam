# Boost special functions
cdef extern from "<boost/math/special_functions.hpp>" namespace "boost::math":
    cpdef double asinh(double x) except +
    cpdef double acosh(double x) except +
    cpdef double atanh(double x) except +
    cpdef double beta(double a, double b) except +
    cdef double _incBeta "boost::math::beta"(double a, double b, double x) except +
    cpdef double gamma "boost::math::tgamma"(double x) except +
    cpdef double digamma(double x) except +
    cpdef double binomial_coefficient[double](unsigned int n, unsigned int k) except +

# Wrapper to fix ordering not matching other functions.
cpdef double incBeta(double x, double a, double b)

cdef extern from "sam.h":
    cdef cppclass CppSam "Sam":
        Sam(size_t, double (*)(double*))
        Sam()
        void setRecordOptions(bint, bint, bint)
        void run(size_t, double*, size_t, size_t)
        double* getSamples()
        # void write(std::string, bool, std::string)
        void addMetropolis(double*, size_t, size_t)
        double getMean(size_t)
        double getVar(size_t)
        double getStd(size_t)
    # Random number generator
    cdef cppclass RNG:
        RNG()
        RNG(unsigned int)
        double normalRand(double, double)
        double normalPDF(double,double,double)
        double normalLogPDF(double, double, double)
        double uniformRand(double, double)
        double uniformPDF(double,double,double)
        double uniformLogPDF(double,double,double)
        int uniformIntRand(int, int)
        double uniformIntPDF(int,int,int)
        double uniformIntLogPDF(int,int,int)
        double gammaRand(double, double)
        double gammaPDF(double, double, double)
        double gammaLogPDF(double, double, double)
        double invGammaRand(double, double)
        double invGammaPDF(double, double, double)
        double invGammaLogPDF(double, double, double)
        double betaRand(double, double)
        double betaPDF(double, double, double)
        double betaLogPDF(double, double, double)
        int poissonRand(double)
        double poissonPDF(int, double)
        double poissonLogPDF(int, double)
        double exponentialRand(double)
        double exponentialPDF(double, double)
        double exponentialLogPDF(double, double)
        int binomialRand(int, double)
        double binomialPDF(int, int, double)
        double binomialLogPDF(int, int, double)

# Wrapper for the RNG class
cdef class RandomNumberGenerator:
    cdef RNG a
    cpdef double normalRand(self,double mean=?, double std=?)
    cpdef double normalPDF(self,double x, double mean, double std)
    cpdef double normalLogPDF(self,double x, double mean, double std)
    cpdef double uniformRand(self, double xMin, double xMax)
    cpdef double uniformPDF(self, double x, double xMin, double xMax)
    cpdef double uniformLogPDF(self, double x,double xMin,double xMax)
    cpdef int uniformIntRand(self, int xMin, int xMax)
    cpdef double uniformIntPDF(self, int x, int xMin, int xMax)
    cpdef double uniformIntLogPDF(self,int x,int xMin,int xMax)
    cpdef double gammaRand(self,double, double)
    cpdef double gammaPDF(self,double, double, double)
    cpdef double gammaLogPDF(self,double, double, double)
    cpdef double invGammaRand(self,double, double)
    cpdef double invGammaPDF(self,double, double, double)
    cpdef double invGammaLogPDF(self,double, double, double)
    cpdef double betaRand(self,double, double)
    cpdef double betaPDF(self,double, double, double)
    cpdef double betaLogPDF(self,double, double, double)
    cpdef int poissonRand(self,double)
    cpdef double poissonPDF(self,int, double)
    cpdef double poissonLogPDF(self,int, double)
    cpdef double exponentialRand(self,double)
    cpdef double exponentialPDF(self,double, double)
    cpdef double exponentialLogPDF(self,double, double)
    cpdef int binomialRand(self,int, double)
    cpdef double binomialPDF(self,int, int, double)
    cpdef double binomialLogPDF(self,int, int, double)

cdef class Sam:
    cdef CppSam *sam
    cpdef void setRecordOptions(self,bint, bint, bint)
