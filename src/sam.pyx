# distutils: language = c++
include "distributions.pyx"
include "griddy.pyx"
import multiprocessing as mp
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal
import numpy as np
from numpy.linalg import solve
import os
cimport numpy as np

# Special function wrappers
cpdef double incBeta(double x, double a, double b):
    return _incBeta(a,b,x)

# Helper functions
cpdef double getWAIC(logLike, samples):
    l = np.array([logLike(i) for i in samples])
    return -2*(logsumexp(l) - log(len(l)) - np.var(l))

cpdef double getAIC(loglike, samples):
    lMax = max([loglike(i) for i in samples])
    return 2*np.shape(samples)[1] - 2*lMax

cpdef double getBIC(loglike, samples, nPoints):
    lMax = max([loglike(i) for i in samples])
    return log(nPoints)*np.shape(samples)[1] - 2 * lMax

cpdef double[:,:] gpCorr(double[:] x, double[:] xPrime, double l, double sigmaSq):
    # TODO: Generalize to other correlation matrix designs
    cdef Size i, j
    output = np.zeros((len(x),len(xPrime)))
    for i in range(len(x)):
        for j in range(len(xPrime)):
            output[i,j] = exp(-(x[i]-xPrime[j])**2 / (2*l**2))
            if abs((x[i] - xPrime[j])/x[i]) < 1e-10:
                output[i,j] += sigmaSq
    return output


cpdef double gpLogLike(double[:] x, double [:] y, double l, double sigmaSq):
    return multivariate_normal.logpdf(y,np.zeros(len(y)),cov=gpCorr(x,x,l, sigmaSq))


cpdef object gpPredict(double[:] targetX, double[:] x, double[:] y, double l, double sigmaSq):
    k1 = np.asarray(gpCorr(x,x,l,sigmaSq))
    k2 = np.asarray(gpCorr(targetX,x,l,sigmaSq))
    pred = np.matmul(k2,solve(k1,np.asarray(y)[:,np.newaxis])).ravel()
    gpVar = (1+sigmaSq) - np.diag(np.matmul(k2,solve(k1,k2.T)))
    gpVar = np.sqrt(gpVar)
    return pred, gpVar


cdef class Sam:
    cpdef double logProbability(self, double[:] position, double[:] gradient, bint computeGradient):
        if self.pyLogProbability is None:
            raise NotImplementedError("You haven't defined the log probability,"+
                "but the sampler called it.")
        return self.pyLogProbability(np.asarray(position),np.asarray(gradient),computeGradient)

    cdef void sample(self):
        cdef size_t s
        for s in range(self.samplers.size()):
            if self.samplers[s].samplerType == 0:
                self.metropolisStep(
                    self.samplers[s].dStart,
                    self.samplers[s].dStop)
            elif self.samplers[s].samplerType == 1:
                self.hmcStep(
                    self.samplers[s].nSteps,
                    self.samplers[s].stepSize,
                    self.samplers[s].dStart,
                    self.samplers[s].dStop)
        return

    cdef void hmcStep(self,Size nSteps, double stepSize, Size dStart, Size dStop):
        cdef Size d, i
        cdef double old
        cdef double new = infinity
        for d in range(self.nDim):
            self.xPropose[d] = self.x[d]
            if d >= dStart and d < dStop:
                self.momentum[d] = normalRand(0,1./sqrt(self.scale[d]))
            else:
                self.momentum[d] = 0

        # Compute the kinetic energy part of the initial Hamiltonian
        cdef double kinetic = 0
        for d in range(dStart,dStop):
            kinetic += self.momentum[d]*self.momentum[d]*self.scale[d] / 2.0

        # Simulate the trajectory
        old = self.logProbability(self.xPropose,self.gradient,True)
        for i in range(nSteps):
            for d in range(dStart,dStop):
                self.momentum[d] += stepSize * self.gradient[d] / 2.0
            self.bouncingMove(stepSize, dStart, dStop)
            new = self.logProbability(self.xPropose, self.gradient,True)
            for d in range(dStart,dStop):
                self.momentum[d] += stepSize * self.gradient[d] / 2.0

        # Compute the kinetic energy part of the proposal Hamiltonian
        cdef double kineticPropose = 0
        for d in range(dStart,dStop):
            kineticPropose += self.momentum[d]*self.momentum[d]*self.scale[d]/2.0

        # Decide whether to accept the new point
        if isnan(new) or isnan(old):
            raise ValueError("Got NaN for the log probability!")
        if (exponentialRand(1.) > old - kinetic - new + kineticPropose):
            for d in range(dStart,dStop):
                self.acceptedView[d] += 1
                self.x[d] = self.xPropose[d]
        return

    cdef void bouncingMove(self, double stepSize, Size dStart, Size dStop):
        cdef Size d
        for d in range(dStart,dStop):
            self.xPropose[d] += self.momentum[d] * stepSize * self.scale[d]
            # Enforce boundary conditions
            while self.hasBoundaries:
                if self.xPropose[d] >= self.upperBoundaries[d]:
                    self.xPropose[d] = 2*self.upperBoundaries[d] - self.xPropose[d]
                    self.momentum[d] = -self.momentum[d]
                    continue
                if self.xPropose[d] <= self.lowerBoundaries[d]:
                    self.xPropose[d] = 2*self.lowerBoundaries[d] - self.xPropose[d]
                    self.momentum[d] = -self.momentum[d]
                    continue
                break
        return

    cdef void metropolisStep(self, Size dStart, Size dStop):
        cdef Size d
        if self.hasBoundaries:
            for d in range(dStart,dStop):
                self.xPropose[d] = self.x[d] + normalRand(0,self.scale[d])
                if(self.xPropose[d] > self.upperBoundaries[d] or
                   self.xPropose[d] < self.lowerBoundaries[d]):
                    return
        else:
            for d in range(dStart,dStop):
                self.xPropose[d] = self.x[d] + normalRand(0,self.scale[d])
        for d in range(0,dStart):
            self.xPropose[d] = self.x[d]
        for d in range(dStop,self.nDim):
            self.xPropose[d] = self.x[d]
        if (exponentialRand(1.) > self.logProbability(self.x,self.gradient,False) -
            self.logProbability(self.xPropose,self.gradient,False)):
            for d in range(dStart,dStop):
                self.acceptedView[d] += 1
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
        cdef Size i, nDims, nPoints
        X = np.asmatrix(x1)
        y = np.asmatrix(y1).T
        nPoints = X.shape[0]
        nDims = X.shape[1]
        if output is None:
            output = np.empty(nDims+1,dtype=np.double)
        V = np.linalg.inv(X.T*X)
        beta_hat = V*X.T*y
        cdef double[:] deviation = np.array(y-X*beta_hat)[:,0]
        cdef double sigmasq = 0
        for i in range(nPoints):
            sigmasq += deviation[i]**2
        sigmasq = invGammaRand((nPoints-nDims)/2.0,sigmasq/2.0)
        deviation = np.array(beta_hat + np.linalg.cholesky(V)*sqrt(sigmasq)*np.random.randn(1,beta_hat.shape[0]).T).ravel()
        for i in range(nDims):
            output[i] = deviation[i]
        output[nDims] = sqrt(sigmasq)
        return output


    cpdef void addMetropolis(self,Size dStart, Size dStop):
        assert dStart >= 0 and dStop <= self.nDim
        cdef SamplerData samp
        samp.samplerType = 0
        samp.dStart = dStart
        samp.dStop = dStop
        self.samplers.push_back(samp)

    cpdef void addHMC(self, Size nSteps, double stepSize, Size dStart, Size dStop):
        assert dStart >= 0 and dStop <= self.nDim
        assert nSteps > 0 and stepSize > 0
        cdef SamplerData samp
        samp.samplerType = 1
        samp.nSteps = nSteps
        samp.stepSize = stepSize
        samp.dStart = dStart
        samp.dStop = dStop
        self.samplers.push_back(samp)

    cpdef void printSamplers(self):
        cdef size_t s
        for s in range(self.samplers.size()):
            if self.samplers[s].samplerType == 0:
                print s, "Metropolis ("+str(self.samplers[s].dStart)+":"+str(self.samplers[s].dStop)+")"
            elif self.samplers[s].samplerType == 1:
                print s, "HMC ("+str(self.samplers[s].dStart)+":"+str(self.samplers[s].dStop)+"), ",\
                    self.samplers[s].nSteps, "steps with size", self.samplers[s].stepSize
        return

    cpdef void clearSamplers(self):
        self.samplers.clear()

    cdef void record(self,Size i):
        cdef Size d
        for d in range(self.recordStart,self.recordStop):
            self.sampleView[i,d-self.recordStart] = self.x[d]
        return

    cdef void recordStats(self):
        cdef Size d
        for d in range(self.nDim):
            self.sampleStats[d](self.x[d])
        return

    cpdef object run(self, Size nSamples, object x0, Size burnIn=0, Size thinning=0, Size recordStart=0, Size recordStop=-1, bint collectStats=False, Size threads=1):
        assert nSamples > 0
        assert type(x0) is np.ndarray and x0.shape[-1] == self.nDim
        cdef Size i, j, d
        cdef double vMag, vMagPropose
        self.recordStart = recordStart
        if recordStop < 0:
            recordStop = self.nDim
        self.recordStop = recordStop
        self.accepted[:] = 0
        self.nSamples = nSamples
        self.burnIn = burnIn
        self.thinning = thinning
        self.sampleStats.clear()
        self.collectStats = collectStats
        if collectStats:
            self.sampleStats.resize(self.nDim)
        self.trials = (self.nSamples + self.burnIn) * (self.thinning+1)
        self.readyToRun = True
        if threads > 1:
            p = mp.Pool(threads)
            self.samples, self.accepted = zip(*p.map(self,[x0]*threads))
            self.samples = np.array(self.samples)
            self.accepted = np.array(self.accepted)
            p.terminate()
            return self.samples
        else:
            self(x0)
            return self.samples

    def __call__(self, double[:] x0):
        assert x0.size == self.nDim
        if not self.readyToRun:
            raise RuntimeError("The call function is for internal use only.")
        self.readyToRun = False
        for d in range(self.nDim):
            self.x[d] = x0[d]
        self.samples = np.empty((self.nSamples,self.recordStop-self.recordStart),dtype=np.double)
        self.sampleView = self.samples
        for i in range(self.nSamples+self.burnIn):
            for j in range(self.thinning+1):
                self.sample()
            if i >= self.burnIn: 
                self.record(i-self.burnIn)
                if self.collectStats:
                    self.recordStats()
        return self.samples, self.accepted

    cpdef object getStats(self):
        assert(not self.sampleStats.size(),"Cannot report statistics without having run the sampler!")
        assert(self.collectStats,"Running statistics collection is turned off.")
        cdef Size d
        means = np.empty(self.nDim,dtype=np.double)
        stds = np.empty(self.nDim,dtype=np.double)
        cdef double[:] meansView = means
        cdef double[:] stdsView = stds
        for d in range(self.nDim):
            meansView[d] = mean(self.sampleStats[d])
            stdsView[d] = sqrt(variance(self.sampleStats[d]))
        return (means, stds)

    cpdef object getAcceptance(self):
        assert self.trials > 0
        return self.accepted.astype(np.double)/self.trials

    cpdef object testGradient(self, double[:] x0, double eps=1e-5):
        assert x0.size == self.nDim
        cdef Size d
        for d in range(self.nDim):
            self.x[d] = x0[d]
        cdef double central = self.logProbability(x0,self.gradient,True)
        cdef double estimate
        cdef object output = np.empty(self.nDim,dtype=np.double)
        cdef double[:] outputView = output
        for d in range(self.nDim):
            x0[d] += self.scale[d]*eps
            estimate = (self.logProbability(x0,self.momentum,False) - central)/(self.scale[d]*eps)
            outputView[d] = (estimate-self.gradient[d])/(estimate+self.gradient[d])
            x0[d] -= self.scale[d]*eps
        return output

    cpdef object gradientDescent(self, double[:] x0, double step=.1, double eps=1e-10):
        assert x0.size == self.nDim
        assert step > 0 and eps > 0
        cdef Size d, i
        cdef bint done = False
        cdef double xNew
        for d in range(self.nDim):
            self.x[d] = x0[d]
        while not done:
            self.logProbability(self.x,self.gradient,True)
            done = True
            for d in range(self.nDim):
                xNew = self.x[d] + step*self.scale[d]*self.scale[d]*self.gradient[d]
                if xNew < self.lowerBoundaries[d]:
                    xNew = self.lowerBoundaries[d]
                if xNew > self.upperBoundaries[d]:
                    xNew = self.upperBoundaries[d]
                if abs(xNew - self.x[d]) > abs(eps * self.scale[d]):
                    done = False
                self.x[d] = xNew
        output = np.zeros(self.nDim)
        for d in range(self.nDim):
            output[d] = self.x[d]
        return output

    cpdef object simulatedAnnealing(self, double[:] x0, Size nSteps=200, Size nQuench=200, double T0=5, double width=1.0):
        assert x0.size == self.nDim
        assert nSteps > 0 and T0 > 0 and width > 0
        cdef Size d, i
        cdef double energy, energyPropose, temperature
        for d in range(self.nDim):
            self.x[d] = x0[d]
        energy = self.logProbability(self.x,self.gradient,False)
        for i in range(nSteps):
            temperature = T0*(1. - (<double>i)/(nSteps))
            for d in range(self.nDim):
                self.xPropose[d] = normalRand(self.x[d],width*self.scale[d])
            energyPropose = self.logProbability(self.xPropose,self.gradient,False)
            if exponentialRand(1.) > (energy - energyPropose)/temperature:
                for d in range(self.nDim):
                    self.x[d] = self.xPropose[d]
                    energy = energyPropose
        for i in range(nQuench):
            for d in range(self.nDim):
                self.xPropose[d] = normalRand(self.x[d],width*self.scale[d]/5.)
            energyPropose = self.logProbability(self.xPropose,self.gradient,False)
            if (energyPropose > energy):
                for d in range(self.nDim):
                    self.x[d] = self.xPropose[d]
                    energy = energyPropose
        output = np.zeros(self.nDim)
        for d in range(self.nDim):
            output[d] = self.x[d]
        return output

    cdef void _setMemoryViews_(self):
        self.x = self._workingMemory_[0:self.nDim]
        self.xPropose = self._workingMemory_[self.nDim:2*self.nDim]
        self.momentum = self._workingMemory_[2*self.nDim:3*self.nDim]
        self.gradient = self._workingMemory_[3*self.nDim:4*self.nDim]
        self.scale = self._workingMemory_[4*self.nDim:5*self.nDim]
        self.upperBoundaries = self._workingMemory_[5*self.nDim:6*self.nDim]
        self.lowerBoundaries = self._workingMemory_[6*self.nDim:7*self.nDim]
        self.acceptedView = self.accepted
        return

    def __init__(self, object logProbability, Size nDim, double[:] scale, double[:] upperBoundaries=None, double[:] lowerBoundaries=None):
        assert scale.size == nDim
        assert logProbability is None or callable(logProbability)
        if upperBoundaries is not None:
            assert upperBoundaries.size == nDim
        if lowerBoundaries is not None:
            assert lowerBoundaries.size == nDim
        cdef Size d
        self.nDim = nDim
        self.readyToRun = False
        self._workingMemory_ = np.empty(7*self.nDim,dtype=np.double)
        self.accepted = np.zeros(self.nDim,dtype=np.intc)
        self.trials = 0
        self._setMemoryViews_()
        self.pyLogProbability = logProbability
        for d in range(self.nDim):
            self.scale[d] = scale[d]
        self.hasBoundaries = False
        if upperBoundaries is not None:
            self.hasBoundaries = True
            for d in range(self.nDim):
                self.upperBoundaries[d] = upperBoundaries[d]
        else:
            for d in range(self.nDim):
                self.upperBoundaries[d] = infinity
        if lowerBoundaries is not None:
            self.hasBoundaries = True
            for d in range(self.nDim):
                self.lowerBoundaries[d] = lowerBoundaries[d]
        else:
            for d in range(self.nDim):
                self.lowerBoundaries[d] = -infinity
        self.extraInitialization()
        return

    def __getstate__(self):
        info = (self.nDim, self.nSamples, self.burnIn, self.thinning, self.recordStart,
                self.recordStop, self.collectStats, self.readyToRun, self.samplers,
                self._workingMemory_, self.accepted, self.pyLogProbability, self.hasBoundaries)
        return info

    def __setstate__(self,info):
        (self.nDim, self.nSamples, self.burnIn, self.thinning, self.recordStart,
         self.recordStop, self.collectStats, self.readyToRun, self.samplers,
         self._workingMemory_, self.accepted, self.pyLogProbability, self.hasBoundaries) = info
        defaultEngine.setSeed(<unsigned long int>int(os.urandom(4).encode("hex"),16))
        np.random.seed(int(os.urandom(4).encode("hex"),16))
        self._setMemoryViews_()
        self.extraInitialization()
        return

    cdef void extraInitialization(self):
        return
