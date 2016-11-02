# distutils: language = c++
# distutils: sources = src/sam.cpp

# Special function wrappers
cpdef double incBeta(double x, double a, double b):
    return _incBeta(a,b,x)

# Wrapper for the RNG class
cdef class RandomNumberGenerator:
    cpdef double normalRand(self, double mean=0, double std=1):
        return self.a.normalRand(mean,std)
    cpdef double normalPDF(self, double x,double mean, double std):
        return self.a.normalPDF(x,mean,std)
    cpdef double normalLogPDF(self, double x, double mean, double std):
        return self.a.normalLogPDF(x,mean,std)
    cpdef double uniformRand(self, double xMin, double xMax):
        return self.a.uniformRand(xMin,xMax)
    cpdef double uniformPDF(self, double x, double xMin,double xMax):
        return self.a.uniformPDF(x,xMin,xMax)
    cpdef double uniformLogPDF(self, double x, double xMin, double xMax):
        return self.a.uniformLogPDF(x,xMin,xMax)
    cpdef int uniformIntRand(self, int xMin, int xMax):
        return self.a.uniformIntRand(xMin,xMax)
    cpdef double uniformIntPDF(self, int x, int xMin, int xMax):
        return self.a.uniformIntPDF(x,xMin,xMax)
    cpdef double uniformIntLogPDF(self, int x, int xMin, int xMax):
        return self.a.uniformIntLogPDF(x,xMin,xMax)
    cpdef double gammaRand(self, double alpha, double beta):
        return self.a.gammaRand(alpha,beta)
    cpdef double gammaPDF(self, double x, double alpha, double beta):
        return self.a.gammaPDF(x,alpha,beta)
    cpdef double gammaLogPDF(self, double x, double alpha, double beta):
        return self.a.gammaLogPDF(x,alpha,beta)
    cpdef double invGammaRand(self, double alpha, double beta):
        return self.a.invGammaRand(alpha,beta)
    cpdef double invGammaPDF(self, double x, double alpha, double beta):
        return self.a.invGammaPDF(x,alpha,beta)
    cpdef double invGammaLogPDF(self, double x, double alpha, double beta):
        return self.a.invGammaLogPDF(x,alpha,beta)
    cpdef double betaRand(self, double alpha, double beta):
        return self.a.betaRand(alpha,beta)
    cpdef double betaPDF(self, double x, double alpha, double beta):
        return self.a.betaPDF(x,alpha,beta)
    cpdef double betaLogPDF(self, double x, double alpha, double beta):
        return self.a.betaLogPDF(x,alpha,beta)
    cpdef int poissonRand(self, double rate):
        return self.a.poissonRand(rate)
    cpdef double poissonPDF(self, int x, double rate):
        return self.a.poissonPDF(x,rate)
    cpdef double poissonLogPDF(self, int x, double rate):
        return self.a.poissonLogPDF(x,rate)
    cpdef double exponentialRand(self, double rate):
        return self.a.exponentialRand(rate)
    cpdef double exponentialPDF(self, double x, double rate):
        return self.a.exponentialPDF(x,rate)
    cpdef double exponentialLogPDF(self, double x, double rate):
        return self.a.exponentialLogPDF(x,rate)
    cpdef int binomialRand(self, int n, double probability):
        return self.a.binomialRand(n,probability)
    cpdef double binomialPDF(self, int x, int n, double probability):
        return self.a.binomialPDF(x,n,probability)
    cpdef double binomialLogPDF(self, int x, int n, double probability):
        return self.a.binomialLogPDF(x,n,probability)

cdef class Sam:
    cpdef void setRecordOptions(self,bint recordSamples, bint accumulateStats, bint printSamples):
        self.sam[0].setRecordOptions(recordSamples, accumulateStats, printSamples)
        return
