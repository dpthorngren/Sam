import numpy as np
cimport numpy as np
from scipy.stats import multivariate_normal
include "distributions.pyx"

cdef class HMCSampler:
    cpdef double logProbability(self, double[:] position):
        if self._testMode == 1:
            return - (position[0] - 4.0)**2/4.0 - (position[1] - 3.0)**2/2.0
        raise NotImplementedError("You haven't defined the log probability,"+
                                  "but the sampler called it.")

    cpdef void gradLogProbability(self, double[:] position, double[:] output):
        if self._testMode == 1:
            output[0] = (position[0]-4.0)/2.0
            output[1] = (position[1]-3.0)/1.0
            return
        raise NotImplementedError("You haven't defined the log probability "+
                                  "gradient, but the sampler called it.")

    cdef void sample(self):
        self.hmcStep(<int>UniformRand(5,100),UniformRand(.01,.5))
        self.metropolisStep(self.scale)
        return

    cdef void hmcStep(self,Size nSteps, double stepSize, int ID=1):
        cdef Size d, i
        # Initialize velocities
        for d in range(self.nDim):
            if self.samplerChoice[d] == ID:
                self.momentum[d] = NormalRand(0,1./sqrt(self.scale[d]))
                self.xPropose[d] = self.x[d]

        # Compute the kinetic energy part of the initial Hamiltonian
        cdef double kinetic = 0
        for d in range(self.nDim):
            if self.samplerChoice[d] == ID:
                kinetic += self.momentum[d]*self.momentum[d]*self.scale[d] / 2.0

        # Simulate the trajectory
        self.gradLogProbability(self.xPropose,self.gradient)
        for i in range(nSteps):
            for d in range(self.nDim):
                if self.samplerChoice[d] == ID:
                    self.momentum[d] += stepSize * self.gradient[d] / 2.0
                    self.xPropose[d] += self.momentum[d] * stepSize * self.scale[d]
            self.gradLogProbability(self.xPropose,self.gradient)
            for d in range(self.nDim):
                self.momentum[d] += stepSize * self.gradient[d] / 2.0

        # Compute the kinetic energy part of the proposal Hamiltonian
        cdef double kineticPropose = 0
        for d in range(self.nDim):
            if self.samplerChoice[d] == ID:
                kineticPropose += self.momentum[d]*self.momentum[d]*self.scale[d]/2.0

        # Decide whether to accept the new point
        if (log(UniformRand()) <
            self.logProbability(self.xPropose) - kineticPropose -
            self.logProbability(self.x) + kinetic):
            for d in range(self.nDim):
                if self.samplerChoice[d] == ID:
                    self.x[d] = self.xPropose[d]
        return

    cdef void metropolisStep(self, double[:] proposalStd, int ID=2):
        cdef Size d
        for d in range(self.nDim):
            if self.samplerChoice[d] == ID:
                self.xPropose[d] = self.x[d] + NormalRand(0,proposalStd[d])
        if (log(UniformRand()) < self.logProbability(self.xPropose) -
            self.logProbability(self.x)):
            for d in range(self.nDim):
                if self.samplerChoice[d] == ID:
                    self.x[d] = self.xPropose[d]
        return

    # TODO: remove need for numpy
    cdef double[:] regressionStep(self, double[:,:] x1, double[:] y1, double[:] output=None):
        '''Computes a linear regression with normal errors on x,y.
        x - The design matrix: columns of predictor variables stacked
            horizontally into a matrix.
        y - An array of variables to be fitted to.
        output - A memoryview to write the resulting samples to.  Should be
            size x1.shape[1] + 1, one for each coefficient plus the standard
            deviation of the result.
        '''
        cdef Size d, dim
        dim = x1.shape[1]
        x = np.asarray(x1)
        y = np.asarray(y1)
        if output is None:
            output = np.empty_like(x[0])
        V = np.linalg.inv(x.T.dot(x))
        beta_hat = V.dot(x.T).dot(y[:,np.newaxis])
        ssd = (y-x.dot(beta_hat)[:,0]).T.dot(y-x.dot(beta_hat)[:,0])
        sigmasq = 1./np.random.gamma((len(y)-np.shape(x)[1])/2.0,2.0/ssd)
        coeffs = (multivariate_normal.rvs(np.zeros(np.shape(x)[1]),V) *
                  np.sqrt(sigmasq[:,np.newaxis]) + beta_hat[:,0])
        for d in range(dim):
            output[d] = coeffs[d]
        output[dim] = sqrt(sigmasq)
        return output

    cdef void record(self,Size i):
        cdef Size d
        for d in range(self.nDim):
            self.sampleView[i,d] = self.x[d]
        return

    cpdef object run(self, Size nSamples, double[:] x0, Size burnIn=0, Size thinning=0):
        '''
            Run the HMC using the given distribution and gradient.
            Arguments:
                nSamples (int): How many samples to take.
                nSteps (int): How many steps to take during integration.
                x0 (double[:]): The starting position of the sampler.
        '''
        cdef Size i, j, d
        cdef double vMag, vMagPropose
        self.samples = np.empty((nSamples,self.nDim),dtype=np.double)
        self.sampleView = self.samples
        for d in range(self.nDim):
            self.x[d] = x0[d]
        for i in range(nSamples+burnIn):
            for j in range(thinning+1):
                self.sample()
            if i >= burnIn: 
                self.record(i-burnIn)
                # TODO: track statistics here
        return self.samples


    def __init__(self,Size nDim, double[:] scale, int[:] samplerChoice=None):
        # TODO: Better documentation
        '''
        Prepares the HMC sampler.
        Arguments:
            nDim: Number of dimensions to the probability distribution (int).
        '''
        cdef Size d
        self.nDim = nDim
        self._testMode = 0
        self.x = np.zeros(self.nDim,dtype=np.double)
        self.momentum = np.empty(self.nDim,dtype=np.double)
        self.xPropose = np.zeros(self.nDim,dtype=np.double)
        self.gradient = np.empty(self.nDim,dtype=np.double)
        self.scale = np.empty(self.nDim,dtype=np.double)
        self.samplerChoice = np.ones(self.nDim,dtype=np.intc)
        for d in range(self.nDim):
            self.scale[d] = scale[d]
        if samplerChoice is not None:
            for d in range(self.nDim):
                self.samplerChoice[d] = samplerChoice[d]
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
    subTest("UniformMean",UniformMean(2,4),3.)
    subTest("UniformVar",UniformVar(2,4),4./12.)
    subTest("UniformStd",UniformStd(2,4),2./sqrt(12.))
    subTest("UniformPDF",UniformPDF(3,2,4),0.5)
    subTest("UniformLPDF",UniformLogPDF(3,2,4),log(0.5))
    subTest("UniformCDF",UniformCDF(2.5,2,4),0.25)
    a = [UniformRand(3,4) for i in range(100000)]
    subTest("UniformRand",np.mean(a),3.5,.01)
    # Normal distribution
    subTest("NormalMean",NormalMean(3,4),3.)
    subTest("NormalVar",NormalVar(3,4),16.)
    subTest("NormalStd",NormalStd(3,4),4.)
    subTest("NormalPDF",NormalPDF(1,3,4),0.08801633)
    subTest("NormalLPDF",NormalLogPDF(1,3,4),log(0.08801633))
    subTest("NormalCDF",NormalCDF(1,3,4),0.30853754)
    a = [NormalRand(3,4) for i in range(100000)]
    subTest("NormRand",np.mean(a),3.,.01)
    # Gamma distribution
    subTest("GammaMean",GammaMean(3,4),.75)
    subTest("GammaVar",GammaVar(3,4),3./16)
    subTest("GammaStd",GammaStd(3,4),sqrt(3)/4.)
    subTest("GammaPDF",GammaPDF(1,3,4),.5861004)
    subTest("GammaLPDF",GammaLogPDF(1,3,4),log(.5861004))
    subTest("GammaCDF",GammaCDF(1,3,4),.7618966)
    a = [GammaRand(3,4) for i in range(100000)]
    subTest("GammaRand",np.mean(a),3./4,.01)
    # InvGamma distribution
    subTest("InvGammaMean",InvGammaMean(3,4),2.)
    subTest("InvGammaVar",InvGammaVar(3,4),4.)
    subTest("InvGammaStd",InvGammaStd(3,4),2.)
    subTest("InvGammaPDF",InvGammaPDF(1,3,4),.006084)
    subTest("InvGammaLPDF",InvGammaLogPDF(1,3,4),log(.006084))
    subTest("InvGammaCDF",InvGammaCDF(1,3,4),.002161,.001)
    a = [InvGammaRand(3,4) for i in range(100000)]
    subTest("InvGammaRand",np.mean(a),2.,.01)
    # Beta distribution
    subTest("BetaMean",BetaMean(3,4),3./7)
    subTest("BetaVar",BetaVar(3,4),.0306122)
    subTest("BetaStd",BetaStd(3,4),0.17496355305)
    subTest("BetaPDF",BetaPDF(.5,3,4),1.875)
    subTest("BetaLPDF",BetaLogPDF(.5,3,4),log(1.875))
    subTest("BetaCDF",BetaCDF(.5,3,4),.65625)
    a = [BetaRand(3,4) for i in range(100000)]
    subTest("BetaRand",np.mean(a),3./7,.01)
    # Poisson distribution
    subTest("PoissonMean",PoissonMean(2.4),2.4)
    subTest("PoissonVar",PoissonVar(2.4),2.4)
    subTest("PoissonStd",PoissonStd(2.4),sqrt(2.4))
    subTest("PoissonPDF",PoissonPDF(3,2.4),.209014)
    subTest("PoissonLPDF",PoissonLogPDF(3,2.4),log(.209014))
    subTest("PoissonCDF",PoissonCDF(3.2,2.4),0.7787229)
    a = [PoissonRand(3.4) for i in range(100000)]
    subTest("PoissonRand",np.mean(a),3.4,.01)
    # Exponential distribution
    subTest("ExpMean",ExponentialMean(2.4),1./2.4)
    subTest("ExpVar",ExponentialVar(2.4),2.4**-2)
    subTest("ExpStd",ExponentialStd(2.4),1./2.4)
    subTest("ExpPDF",ExponentialPDF(1,2.4),.217723)
    subTest("ExpLPDF",ExponentialLogPDF(1,2.4),log(.217723))
    subTest("ExpCDF",ExponentialCDF(1,2.4),0.9092820)
    a = [ExponentialRand(3.4) for i in range(100000)]
    subTest("ExpRand",np.mean(a),1./3.4,.01)
    # Binomial distribution
    subTest("BinMean",BinomialMean(10,.4),4.)
    subTest("BinVar",BinomialVar(10,.4),.4*.6*10.)
    subTest("BinStd",BinomialStd(10,.4),sqrt(.4*.6*10.))
    subTest("BinPDF",BinomialPDF(3,10,.4),.2149908)
    subTest("BinLPDF",BinomialLogPDF(3,10,.4),log(.2149908))
    subTest("BinCDF",BinomialCDF(3.4,10,.4),0.3822806)
    a = [BinomialRand(10,.74) for i in range(100000)]
    subTest("BinRand",np.mean(a),7.4,.01)
    return
