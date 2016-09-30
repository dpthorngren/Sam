include "distributions.pyx"
import numpy as np
import time
cimport numpy as np

Uniform = _uniform(<unsigned long int>(1*1000*time.time()))
Normal = _normal(<unsigned long int>(2*1000*time.time()))
Gamma = _gamma(<unsigned long int>(3*1000*time.time()))
InvGamma = _invGamma(<unsigned long int>(4*1000*time.time()))
Beta = _beta(<unsigned long int>(5*1000*time.time()))
Poisson = _poisson(<unsigned long int>(6*1000*time.time()))
Exponential = _expon(<unsigned long int>(7*1000*time.time()))
Binomial = _binomial(<unsigned long int>(7*1000*time.time()))


cdef class HMCSampler:
    '''
    Prepares the HMC sampler.
    Arguments:
        nDim: Number of dimensions to the probability distribution (int).
        stepSize: The length of a step in the  integrator (double).
    '''
    cpdef double logProbability(self, double[:] position):
        return - (position[0] - 4.0)**2/4.0 - (position[1] - 3.0)**2/2.0

    cpdef void gradLogProbability(self, double[:] position, double[:] output):
        output[0] = (position[0]-4.0)/2.0
        output[1] = (position[1]-3.0)/1.0
        return

    cdef void simTrajectory(self, Size nSteps):
        '''
            Computes a path taken by the particle from its current
            position and velocity, given the potential -logProbability.
            Argument:
                nSteps (int): How many steps to take before stopping.
        '''
        cdef Size i, d
        self.gradLogProbability(self.xPropose,self.gradient)
        for i in range(nSteps):
            for d in range(self.nDim):
                # TODO: double step
                self.vPropose[d] -= self.stepSize * self.gradient[d] / 2.0
                self.xPropose[d] += self.vPropose[d] * self.stepSize
            self.gradLogProbability(self.xPropose,self.gradient)
            for d in range(self.nDim):
                self.vPropose[d] -= self.stepSize * self.gradient[d] / 2.0
        return

    cpdef object run(self, Size nSamples, Size nSteps, double[:] x0,Size burnIn=0):
        '''
            Run the HMC using the given distribution and gradient.
            Arguments:
                nSamples (int): How many samples to take.
                nSteps (int): How many steps to take during integration.
                x0 (double[:]): The starting position of the sampler.
        '''
        cdef Size i, d
        cdef double vMag, vMagPropose
        self.samples = np.empty((nSamples,self.nDim),dtype=np.double)
        cdef double[:,:] sampleView = self.samples
        for d in range(self.nDim):
            self.x[d] = x0[d]
        for i in range(nSamples+burnIn):
            for d in range(self.nDim):
                self.vPropose[d] = np.random.randn()
            self.simTrajectory(<int>(nSteps*(np.random.rand()+.5)))
            vMag = 0
            vMagPropose = 0
            for d in range(self.nDim):
                vMag += self.v[d]*self.v[d]
                vMagPropose += self.vPropose[d]*self.vPropose[d]
            if (np.random.rand() <
                np.exp(self.logProbability(self.xPropose) + vMagPropose/2.0 -
                    self.logProbability(self.x) - vMag/2.0)):
                    for d in range(self.nDim):
                        self.x[d] = self.xPropose[d]
            if i >= burnIn: 
                for d in range(self.nDim):
                    sampleView[i-burnIn,d] = self.x[d]
        return self.samples


    cpdef recordTrajectory(self,double[:] x0, double[:] v0, Size nSteps):
        '''
        Given an initial position and velocity, returns the trajectory
        taken by a particle in the potential given by U=-log(probability).
        Not used internally, this function is for debugging the energy
        and gradient functions.

        Inputs:
            x0 - The starting position.  Should have the same length as the
                number of dimensions.
            y0 - The starting velocity.  Should have the same length as the
                number of dimensions.
            nSteps - How many leapfrog steps to take.

        Output:
            A numpy array containing the positions after each step, including
            the starting position.  Dimensions are [nSteps,nDimensions].
        '''
        cdef Size i, d
        for d in range(self.nDim):
            self.xPropose[d] = x0[d]
            self.vPropose[d] = v0[d]
        output = np.empty((nSteps,self.nDim),dtype=np.double)
        for d in range(self.nDim):
            output[0,d] = self.xPropose[d]
        for i in range(nSteps):
            self.simTrajectory(1)
            for d in range(self.nDim):
                output[i,d] = self.xPropose[d]
        return output

    def __init__(self,Size nDim, double stepSize):
        '''
        Prepares the HMC sampler.
        Arguments:
            nDim: Number of dimensions to the probability distribution (int).
            stepSize: The length of a step in the  integrator (double).
        '''
        self.nDim = nDim
        self.stepSize = stepSize
        self.x = np.zeros(self.nDim,dtype=np.double)
        self.v = np.empty(self.nDim,dtype=np.double)
        self.xPropose = np.zeros(self.nDim,dtype=np.double)
        self.vPropose = np.empty(self.nDim,dtype=np.double)
        self.gradient = np.empty(self.nDim,dtype=np.double)
        return


def subTest(name, double a, double b, double prec=.0001):
    # Low standard so I can use only a few significant digits.
    if abs(2.*(a - b)/(a+b)) < prec:
        print " PASS ", "{:15s}".format(name), "{:10f}".format(a), "{:10f}".format(b)
    else:
        print "*FAIL*", "{:15s}".format(name), "{:10f}".format(a), "{:10f}".format(b)
    return

def test():
    '''
    Note that some of these functions are probabilistic,
    and can fail by sheer bad luck.
    '''
    print "===== Testing System ====="
    print "First, the following should fail:"
    subTest("Should Fail",-1,1.00000001)
    print "Now, the following should pass:"
    subTest("Should Pass",8,8.00000001)
    print ""
    print "===== Basic Functions ====="
    print "State  Name              Value      Expected"
    subTest("Sin", sin(3.45),-.303542)
    subTest("Cos", cos(3.45),-.952818)
    subTest("Tan", tan(1.25),3.009569)
    subTest("Arcsin", asin(.15),.150568)
    subTest("Arccos", acos(.15),1.42022)
    subTest("Arctan", atan(.15),.1488899)
    subTest("Sinh", sinh(2.45),5.75103)
    subTest("Cosh", cosh(2.45),5.83732)
    subTest("Tanh", tanh(2.25),.978026)
    subTest("Arcsinh", asinh(3.),1.8184464)
    subTest("Arccosh", acosh(3.),1.7627471)
    subTest("Arctanh", atanh(1/3.),.34657359)
    # subTest("Choose", binomial_coefficient(16,13),560.)
    print ""
    print "===== Special Functions ====="
    print "State  Name              Value      Expected"
    subTest("Beta",beta(.7,2.5),0.711874)
    subTest("Gamma", gamma(2.5),1.32934)
    subTest("Digamma", digamma(12.5),2.48520)
    print ""
    print "===== Distributions ====="
    # TODO: Test gradients
    print "State  Name              Value      Expected"
    # Uniform distribution
    subTest("UniformMean",Uniform.mean(2,4),3.)
    subTest("UniformVar",Uniform.var(2,4),4./12.)
    subTest("UniformStd",Uniform.std(2,4),2./sqrt(12.))
    subTest("UniformPDF",Uniform.pdf(3,2,4),0.5)
    subTest("UniformLPDF",Uniform.logPDF(3,2,4),log(0.5))
    subTest("UniformCDF",Uniform.cdf(2.5,2,4),0.25)
    a = [Uniform.rand(3,4) for i in range(100000)]
    subTest("UniformRand",np.mean(a),3.5,.01)
    # Normal distribution
    subTest("NormalMean",Normal.mean(3,4),3.)
    subTest("NormalVar",Normal.var(3,4),16.)
    subTest("NormalStd",Normal.std(3,4),4.)
    subTest("NormalPDF",Normal.pdf(1,3,4),0.08801633)
    subTest("NormalLPDF",Normal.logPDF(1,3,4),log(0.08801633))
    subTest("NormalCDF",Normal.cdf(1,3,4),0.30853754)
    a = [Normal.rand(3,4) for i in range(100000)]
    subTest("NormRand",np.mean(a),3.,.01)
    # Gamma distribution
    subTest("GammaMean",Gamma.mean(3,4),.75)
    subTest("GammaVar",Gamma.var(3,4),3./16)
    subTest("GammaStd",Gamma.std(3,4),sqrt(3)/4.)
    subTest("GammaPDF",Gamma.pdf(1,3,4),.5861004)
    subTest("GammaLPDF",Gamma.logPDF(1,3,4),log(.5861004))
    subTest("GammaCDF",Gamma.cdf(1,3,4),.7618966)
    a = [Gamma.rand(3,4) for i in range(100000)]
    subTest("GammaRand",np.mean(a),3./4,.01)
    # InvGamma distribution
    subTest("InvGammaMean",InvGamma.mean(3,4),2.)
    subTest("InvGammaVar",InvGamma.var(3,4),4.)
    subTest("InvGammaStd",InvGamma.std(3,4),2.)
    subTest("InvGammaPDF",InvGamma.pdf(1,3,4),.006084)
    subTest("InvGammaLPDF",InvGamma.logPDF(1,3,4),log(.006084))
    subTest("InvGammaCDF",InvGamma.cdf(1,3,4),.002161,.001)
    a = [InvGamma.rand(3,4) for i in range(100000)]
    subTest("InvGammaRand",np.mean(a),2.,.01)
    # Beta distribution
    subTest("BetaMean",Beta.mean(3,4),3./7)
    subTest("BetaVar",Beta.var(3,4),.0306122)
    subTest("BetaStd",Beta.std(3,4),0.17496355305)
    subTest("BetaPDF",Beta.pdf(.5,3,4),1.875)
    subTest("BetaLPDF",Beta.logPDF(.5,3,4),log(1.875))
    subTest("BetaCDF",Beta.cdf(.5,3,4),.65625)
    a = [Beta.rand(3,4) for i in range(100000)]
    subTest("BetaRand",np.mean(a),3./7,.01)
    # Poisson distribution
    subTest("PoissonMean",Poisson.mean(2.4),2.4)
    subTest("PoissonVar",Poisson.var(2.4),2.4)
    subTest("PoissonStd",Poisson.std(2.4),sqrt(2.4))
    subTest("PoissonPDF",Poisson.pdf(3,2.4),.209014)
    subTest("PoissonLPDF",Poisson.logPDF(3,2.4),log(.209014))
    subTest("PoissonCDF",Poisson.cdf(3.2,2.4),0.7787229)
    a = [Poisson.rand(3.4) for i in range(100000)]
    subTest("PoissonRand",np.mean(a),3.4,.01)
    # Exponential distribution
    subTest("ExpMean",Exponential.mean(2.4),1./2.4)
    subTest("ExpVar",Exponential.var(2.4),2.4**-2)
    subTest("ExpStd",Exponential.std(2.4),1./2.4)
    subTest("ExpPDF",Exponential.pdf(1,2.4),.217723)
    subTest("ExpLPDF",Exponential.logPDF(1,2.4),log(.217723))
    subTest("ExpCDF",Exponential.cdf(1,2.4),0.9092820)
    a = [Exponential.rand(3.4) for i in range(100000)]
    subTest("ExpRand",np.mean(a),1./3.4,.01)
    # Binomial distribution
    subTest("BinMean",Binomial.mean(10,.4),4.)
    subTest("BinVar",Binomial.var(10,.4),.4*.6*10.)
    subTest("BinStd",Binomial.std(10,.4),sqrt(.4*.6*10.))
    subTest("BinPDF",Binomial.pdf(3,10,.4),.2149908)
    subTest("BinLPDF",Binomial.logPDF(3,10,.4),log(.2149908))
    subTest("BinCDF",Binomial.cdf(3.4,10,.4),0.3822806)
    a = [Binomial.rand(10,.74) for i in range(100000)]
    subTest("BinRand",np.mean(a),7.4,.01)
    return
