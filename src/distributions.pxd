cdef extern from "sam.h":
    cdef cppclass RNG:
        RNG()
        RNG(unsigned int)
        double normalMean(double mean, double std) except +
        double normalVar(double mean, double std) except +
        double normalStd(double mean, double std) except +
        double normalRand(double mean, double std) except +
        double normalPDF(double x, double mean, double std) except +
        double normalLogPDF(double x, double mean, double std) except +
        double uniformMean(double xMin, double xMax) except +
        double uniformVar(double xMin, double xMax) except +
        double uniformStd(double xMin, double xMax) except +
        double uniformRand(double xMin, double xMax) except +
        double uniformPDF(double x, double xMin, double xMax) except +
        double uniformLogPDF(double x, double xMin, double xMax) except +
        double uniformCDF(double x, double xMin, double xMax) except +
        double uniformIntMean(int xMin, int xMax) except +
        double uniformIntVar(int xMin, int xMax) except +
        double uniformIntStd(int xMin, int xMax) except +
        int uniformIntRand(int xMin, int xMax) except +
        double uniformIntPDF(int x, int xMin, int xMax) except +
        double uniformIntLogPDF(int x, int xMin, int xMax) except +
        double uniformIntCDF(double x, int xMin, int xMax) except +
        double gammaMean(double shape, double rate) except +
        double gammaVar(double shape, double rate) except +
        double gammaStd(double shape, double rate) except +
        double gammaRand(double shape, double rate) except +
        double gammaPDF(double x, double shape, double rate) except +
        double gammaLogPDF(double x, double shape, double rate) except +
        double gammaCDF(double x, double shape, double rate) except +
        double invGammaMean(double shape, double rate) except +
        double invGammaVar(double shape, double rate) except +
        double invGammaStd(double shape, double rate) except +
        double invGammaRand(double shape, double rate) except +
        double invGammaPDF(double x, double shape, double rate) except +
        double invGammaLogPDF(double x, double shape, double rate) except +
        double invGammaCDF(double x, double shape, double rate) except +
        double betaMean(double alpha, double beta) except +
        double betaVar(double alpha, double beta) except +
        double betaStd(double alpha, double beta) except +
        double betaRand(double alpha, double beta) except +
        double betaPDF(double x, double alpha, double beta) except +
        double betaLogPDF(double x, double alpha, double beta) except +
        double betaCDF(double x, double alpha, double beta) except +
        double poissonMean(double rate) except +
        double poissonVar(double rate) except +
        double poissonStd(double rate) except +
        int poissonRand(double rate) except +
        double poissonPDF(int x, double rate) except +
        double poissonLogPDF(int x, double rate) except +
        double poissonCDF(double x, double rate) except +
        double exponentialMean(double rate) except +
        double exponentialVar(double rate) except +
        double exponentialStd(double rate) except +
        double exponentialRand(double rate) except +
        double exponentialPDF(double x, double rate) except +
        double exponentialLogPDF(double x, double rate) except +
        double exponentialCDF(double x, double rate) except +
        double binomialMean(int number, double probability) except +
        double binomialVar(int number, double probability) except +
        double binomialStd(int number, double probability) except +
        int binomialRand(int number, double probability) except +
        double binomialPDF(int x, int number, double probability) except +
        double binomialLogPDF(int x, int number, double probability) except +
        double binomialCDF(double x, int number, double probability) except +

cdef class RandomNumberGenerator:
    cdef RNG rng
    cdef double _normalMean(self, double mean, double std)
    cpdef object normalMean(self, object mean, object std)
    cdef double _normalVar(self, double mean, double std)
    cpdef object normalVar(self, object mean, object std)
    cdef double _normalStd(self, double mean, double std)
    cpdef object normalStd(self, object mean, object std)
    cdef double _normalRand(self, double mean, double std)
    cpdef object normalRand(self, object mean, object std)
    cdef double _normalPDF(self, double x, double mean, double std)
    cpdef object normalPDF(self, object x, object mean, object std)
    cdef double _normalLogPDF(self, double x, double mean, double std)
    cpdef object normalLogPDF(self, object x, object mean, object std)
    cdef double _uniformMean(self, double xMin, double xMax)
    cpdef object uniformMean(self, object xMin, object xMax)
    cdef double _uniformVar(self, double xMin, double xMax)
    cpdef object uniformVar(self, object xMin, object xMax)
    cdef double _uniformStd(self, double xMin, double xMax)
    cpdef object uniformStd(self, object xMin, object xMax)
    cdef double _uniformRand(self, double xMin, double xMax)
    cpdef object uniformRand(self, object xMin, object xMax)
    cdef double _uniformPDF(self, double x, double xMin, double xMax)
    cpdef object uniformPDF(self, object x, object xMin, object xMax)
    cdef double _uniformLogPDF(self, double x, double xMin, double xMax)
    cpdef object uniformLogPDF(self, object x, object xMin, object xMax)
    cdef double _uniformCDF(self, double x, double xMin, double xMax)
    cpdef object uniformCDF(self, object x, object xMin, object xMax)
    cdef double _uniformIntMean(self, int xMin, int xMax)
    cpdef object uniformIntMean(self, object xMin, object xMax)
    cdef double _uniformIntVar(self, int xMin, int xMax)
    cpdef object uniformIntVar(self, object xMin, object xMax)
    cdef double _uniformIntStd(self, int xMin, int xMax)
    cpdef object uniformIntStd(self, object xMin, object xMax)
    cdef int _uniformIntRand(self, int xMin, int xMax)
    cpdef object uniformIntRand(self, object xMin, object xMax)
    cdef double _uniformIntPDF(self, int x, int xMin, int xMax)
    cpdef object uniformIntPDF(self, object x, object xMin, object xMax)
    cdef double _uniformIntLogPDF(self, int x, int xMin, int xMax)
    cpdef object uniformIntLogPDF(self, object x, object xMin, object xMax)
    cdef double _uniformIntCDF(self, double x, int xMin, int xMax)
    cpdef object uniformIntCDF(self, object x, object xMin, object xMax)
    cdef double _gammaMean(self, double shape, double rate)
    cpdef object gammaMean(self, object shape, object rate)
    cdef double _gammaVar(self, double shape, double rate)
    cpdef object gammaVar(self, object shape, object rate)
    cdef double _gammaStd(self, double shape, double rate)
    cpdef object gammaStd(self, object shape, object rate)
    cdef double _gammaRand(self, double shape, double rate)
    cpdef object gammaRand(self, object shape, object rate)
    cdef double _gammaPDF(self, double x, double shape, double rate)
    cpdef object gammaPDF(self, object x, object shape, object rate)
    cdef double _gammaLogPDF(self, double x, double shape, double rate)
    cpdef object gammaLogPDF(self, object x, object shape, object rate)
    cdef double _gammaCDF(self, double x, double shape, double rate)
    cpdef object gammaCDF(self, object x, object shape, object rate)
    cdef double _invGammaMean(self, double shape, double rate)
    cpdef object invGammaMean(self, object shape, object rate)
    cdef double _invGammaVar(self, double shape, double rate)
    cpdef object invGammaVar(self, object shape, object rate)
    cdef double _invGammaStd(self, double shape, double rate)
    cpdef object invGammaStd(self, object shape, object rate)
    cdef double _invGammaRand(self, double shape, double rate)
    cpdef object invGammaRand(self, object shape, object rate)
    cdef double _invGammaPDF(self, double x, double shape, double rate)
    cpdef object invGammaPDF(self, object x, object shape, object rate)
    cdef double _invGammaLogPDF(self, double x, double shape, double rate)
    cpdef object invGammaLogPDF(self, object x, object shape, object rate)
    cdef double _invGammaCDF(self, double x, double shape, double rate)
    cpdef object invGammaCDF(self, object x, object shape, object rate)
    cdef double _betaMean(self, double alpha, double beta)
    cpdef object betaMean(self, object alpha, object beta)
    cdef double _betaVar(self, double alpha, double beta)
    cpdef object betaVar(self, object alpha, object beta)
    cdef double _betaStd(self, double alpha, double beta)
    cpdef object betaStd(self, object alpha, object beta)
    cdef double _betaRand(self, double alpha, double beta)
    cpdef object betaRand(self, object alpha, object beta)
    cdef double _betaPDF(self, double x, double alpha, double beta)
    cpdef object betaPDF(self, object x, object alpha, object beta)
    cdef double _betaLogPDF(self, double x, double alpha, double beta)
    cpdef object betaLogPDF(self, object x, object alpha, object beta)
    cdef double _betaCDF(self, double x, double alpha, double beta)
    cpdef object betaCDF(self, object x, object alpha, object beta)
    cdef double _poissonMean(self, double rate)
    cpdef object poissonMean(self, object rate)
    cdef double _poissonVar(self, double rate)
    cpdef object poissonVar(self, object rate)
    cdef double _poissonStd(self, double rate)
    cpdef object poissonStd(self, object rate)
    cdef int _poissonRand(self, double rate)
    cpdef object poissonRand(self, object rate)
    cdef double _poissonPDF(self, int x, double rate)
    cpdef object poissonPDF(self, object x, object rate)
    cdef double _poissonLogPDF(self, int x, double rate)
    cpdef object poissonLogPDF(self, object x, object rate)
    cdef double _poissonCDF(self, double x, double rate)
    cpdef object poissonCDF(self, object x, object rate)
    cdef double _exponentialMean(self, double rate)
    cpdef object exponentialMean(self, object rate)
    cdef double _exponentialVar(self, double rate)
    cpdef object exponentialVar(self, object rate)
    cdef double _exponentialStd(self, double rate)
    cpdef object exponentialStd(self, object rate)
    cdef double _exponentialRand(self, double rate)
    cpdef object exponentialRand(self, object rate)
    cdef double _exponentialPDF(self, double x, double rate)
    cpdef object exponentialPDF(self, object x, object rate)
    cdef double _exponentialLogPDF(self, double x, double rate)
    cpdef object exponentialLogPDF(self, object x, object rate)
    cdef double _exponentialCDF(self, double x, double rate)
    cpdef object exponentialCDF(self, object x, object rate)
    cdef double _binomialMean(self, int number, double probability)
    cpdef object binomialMean(self, object number, object probability)
    cdef double _binomialVar(self, int number, double probability)
    cpdef object binomialVar(self, object number, object probability)
    cdef double _binomialStd(self, int number, double probability)
    cpdef object binomialStd(self, object number, object probability)
    cdef int _binomialRand(self, int number, double probability)
    cpdef object binomialRand(self, object number, object probability)
    cdef double _binomialPDF(self, int x, int number, double probability)
    cpdef object binomialPDF(self, object x, object number, object probability)
    cdef double _binomialLogPDF(self, int x, int number, double probability)
    cpdef object binomialLogPDF(self, object x, object number, object probability)
    cdef double _binomialCDF(self, double x, int number, double probability)
    cpdef object binomialCDF(self, object x, object number, object probability)
    cdef object wrapDD(self, double (*func)(RandomNumberGenerator, double), object arg1)
    cdef object wrapDDD(self, double (*func)(RandomNumberGenerator, double, double), object arg1, object arg2)
    cdef object wrapDDDD(self, double (*func)(RandomNumberGenerator, double, double, double), object arg1, object arg2, object arg3)
    cdef object wrapDDID(self, double (*func)(RandomNumberGenerator, double, int, double), object arg1, object arg2, object arg3)
    cdef object wrapDDII(self, double (*func)(RandomNumberGenerator, double, int, int), object arg1, object arg2, object arg3)
    cdef object wrapDID(self, double (*func)(RandomNumberGenerator, int, double), object arg1, object arg2)
    cdef object wrapDII(self, double (*func)(RandomNumberGenerator, int, int), object arg1, object arg2)
    cdef object wrapDIID(self, double (*func)(RandomNumberGenerator, int, int, double), object arg1, object arg2, object arg3)
    cdef object wrapDIII(self, double (*func)(RandomNumberGenerator, int, int, int), object arg1, object arg2, object arg3)
    cdef object wrapID(self, int (*func)(RandomNumberGenerator, double), object arg1)
    cdef object wrapIID(self, int (*func)(RandomNumberGenerator, int, double), object arg1, object arg2)
    cdef object wrapIII(self, int (*func)(RandomNumberGenerator, int, int), object arg1, object arg2)
