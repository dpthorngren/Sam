# distutils: language = c++
include "distributions.pyx"
include "griddy.pyx"
import multiprocessing as mp
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal
from scipy.linalg import solve_triangular
from sys import stdout
import numpy as np
from numpy.linalg import solve, cholesky
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

def gpGaussKernel(x,xPrime,theta):
    return theta[1]*np.exp(-abs(x[:,np.newaxis]-xPrime[np.newaxis,:])**2/(2*theta[0]**2))

def gpExpKernel(x,xPrime,theta):
    return theta[1]*np.exp(-abs(x[:,np.newaxis]-xPrime[np.newaxis,:])/theta[0])

def gaussianProcess(x, y, theta, xTest=None, kernel=gpExpKernel, kernelChol=None):
    if kernelChol is None:
        K = kernel(x,x,theta) + theta[2]*np.eye(len(x))
        L = cholesky(K)
    else:
        L = kernelChol
    alpha = solve_triangular(L.T,solve_triangular(L,y,lower=True))
    if xTest is not None:
        KTest = kernel(x,xTest,theta)
        v = solve_triangular(L,KTest,lower=True)
        predVariance = kernel(xTest,xTest,theta) + np.eye(len(xTest))*theta[2] - np.matmul(v.T,v)
        return np.matmul(KTest.T,alpha), predVariance
    return -.5*np.sum(y*alpha) - np.sum(np.log(np.diag(L)))

cdef class Sam:
    cpdef double logProbability(self, double[:] position, double[:] gradient, bint computeGradient):
        if self.pyLogProbability is None:
            raise NotImplementedError("You haven't defined the log probability,"+
                "but the sampler called it.")
        return self.pyLogProbability(np.asarray(position),np.asarray(gradient),computeGradient)

    cdef void sample(self):
        cdef size_t s
        cdef double logP0 = nan
        for s in range(self.samplers.size()):
            if self.samplers[s].samplerType == 0:
                logP0 = self.metropolisStep(
                    self.samplers[s].dStart,
                    self.samplers[s].dStop, logP0)
            elif self.samplers[s].samplerType == 1:
                logP0 = self.hmcStep(
                    self.samplers[s].nSteps,
                    self.samplers[s].stepSize,
                    self.samplers[s].dStart,
                    self.samplers[s].dStop, logP0)
        return

    cdef double hmcStep(self,Size nSteps, double stepSize, Size dStart, Size dStop, double logP0=nan) except +:
        cdef Size d, i
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
        if isnan(logP0):
            logP0 = self.logProbability(self.xPropose,self.gradient,True)
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
        if isnan(new) or isnan(logP0):
            raise ValueError("Got NaN for the log probability!")
        if (exponentialRand(1.) > logP0 - kinetic - new + kineticPropose):
            for d in range(dStart,dStop):
                self.acceptedView[d] += 1
                self.x[d] = self.xPropose[d]
            return new
        return logP0

    cdef void bouncingMove(self, double stepSize, Size dStart, Size dStop) except +:
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

    cdef double metropolisStep(self, Size dStart, Size dStop, double logP0 = nan) except +:
        cdef Size d
        cdef double logP1
        for d in range(0,self.nDim):
            if d >= dStart and d < dStop:
                self.xPropose[d] = self.x[d] + normalRand(0,self.scale[d])
                if self.hasBoundaries and (self.xPropose[d] > self.upperBoundaries[d] or
                   self.xPropose[d] < self.lowerBoundaries[d]):
                    return logP0
            else:
                self.xPropose[d] = self.x[d]
        if isnan(logP0):
            logP0 = self.logProbability(self.x,self.gradient,False)
        logP1 = self.logProbability(self.xPropose,self.gradient,False)
        if (exponentialRand(1.) > logP0 - logP1):
            for d in range(dStart,dStop):
                self.acceptedView[d] += 1
                self.x[d] = self.xPropose[d]
            return logP1
        return logP0

    cdef double metropolisCorrStep(self, Size dStart, Size dStop, double[:,:] proposeChol, double logP0 = nan) except +:
        cdef Size d
        cdef double logP1
        mvNormalRand(self.x[dStart:],proposeChol,self.xPropose[dStart:],True)
        for d in range(0,self.nDim):
            if d >= dStart and d < dStop:
                if self.hasBoundaries and (self.xPropose[d] > self.upperBoundaries[d] or
                   self.xPropose[d] < self.lowerBoundaries[d]):
                    return logP0
            else:
                self.xPropose[d] = self.x[d]
        if isnan(logP0):
            logP0 = self.logProbability(self.x,self.gradient,False)
        logP1 = self.logProbability(self.xPropose,self.gradient,False)
        if (exponentialRand(1.) > logP0 - logP1):
            for d in range(dStart,dStop):
                self.acceptedView[d] += 1
                self.x[d] = self.xPropose[d]
            return logP1
        return logP0

    # TODO: remove need for numpy
    cdef double[:] regressionStep(self, double[:,:] x1, double[:] y1, double[:] output=None) except +:
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


    cpdef void addMetropolis(self,Size dStart, Size dStop) except +:
        assert dStart >= 0 and dStart < self.nDim, "The start dimension must be between 0 and nDim - 1 (inclusive)."
        assert dStop > 0 and dStop <= self.nDim, "The stop dimension must be between 1 and nDim (inclusive)."
        cdef SamplerData samp
        samp.samplerType = 0
        samp.dStart = dStart
        samp.dStop = dStop
        self.samplers.push_back(samp)

    cpdef void addHMC(self, Size nSteps, double stepSize, Size dStart, Size dStop) except +:
        assert dStart >= 0 and dStart < self.nDim, "The start dimension must be between 0 and nDim - 1 (inclusive)."
        assert dStop > 0 and dStop <= self.nDim, "The stop dimension must be between 1 and nDim (inclusive)."
        assert nSteps > 0 and stepSize > 0, "The step size and the number of steps must be greater than zero."
        cdef SamplerData samp
        samp.samplerType = 1
        samp.nSteps = nSteps
        samp.stepSize = stepSize
        samp.dStart = dStart
        samp.dStop = dStop
        self.samplers.push_back(samp)

    cpdef void printSamplers(self) except +:
        cdef size_t s
        for s in range(self.samplers.size()):
            if self.samplers[s].samplerType == 0:
                print s, "Metropolis ("+str(self.samplers[s].dStart)+":"+str(self.samplers[s].dStop)+")"
            elif self.samplers[s].samplerType == 1:
                print s, "HMC ("+str(self.samplers[s].dStart)+":"+str(self.samplers[s].dStop)+"), ",\
                    self.samplers[s].nSteps, "steps with size", self.samplers[s].stepSize
        return

    cpdef void clearSamplers(self) except +:
        self.samplers.clear()

    cdef void record(self,Size i) except +:
        cdef Size d
        for d in range(self.recordStart,self.recordStop):
            self.sampleView[i,d-self.recordStart] = self.x[d]
        return

    cdef void recordStats(self) except +:
        cdef Size d
        for d in range(self.nDim):
            self.sampleStats[d](self.x[d])
        return

    cpdef object run(self, Size nSamples, object x0, Size burnIn=0, Size thinning=0, Size recordStart=0, Size recordStop=-1, bint collectStats=False, Size threads=1, bint showProgress=True) except +:
        assert nSamples > 0, "The number of samples must be greater than 0."
        assert type(x0) is np.ndarray, "The initial position must be an array."
        assert x0.shape[-1] == self.nDim, "The initial position has the wrong number of dimensions."
        cdef Size i, j, d
        cdef double vMag, vMagPropose
        self.showProgress = showProgress
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
        assert x0.size == self.nDim, "The initial position has the wrong number of dimensions."
        if not self.readyToRun:
            raise RuntimeError("The call function is for internal use only.")
        self.readyToRun = False
        for d in range(self.nDim):
            self.x[d] = x0[d]
        self.samples = np.empty((self.nSamples,self.recordStop-self.recordStart),dtype=np.double)
        self.sampleView = self.samples
        cdef Size totalDraws = self.nSamples + self.burnIn
        for i in range(totalDraws):
            for j in range(self.thinning+1):
                self.sample()
            if i >= self.burnIn: 
                self.record(i-self.burnIn)
                if self.collectStats:
                    self.recordStats()
            if self.showProgress and i%(totalDraws/100+1)==0:
                if i < self.burnIn:
                    self.progressBar(i+1,self.burnIn,"Burning-in")
                else:
                    self.progressBar(i+1-self.burnIn,self.nSamples,"Sampling")
        if self.showProgress:
            self.progressBar(i+1-self.burnIn,self.nSamples,"Sampling")
        return self.samples, self.accepted

    cpdef object getStats(self) except +:
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

    cpdef object getAcceptance(self) except +:
        assert self.trials > 0, "The number of trials must be greater than zero to compute the acceptance rate."
        return self.accepted.astype(np.double)/self.trials

    cpdef object testGradient(self, double[:] x0, double eps=1e-5) except +:
        assert x0.size == self.nDim, "The starting position given has wrong number of dimensions."
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

    cpdef object gradientDescent(self, double[:] x0, double step=.1, double eps=1e-10) except +:
        assert x0.size == self.nDim, "The starting position given has wrong number of dimensions."
        assert step > 0 and eps > 0, "Both the step size and the error bound must be positive."
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

    cpdef object simulatedAnnealing(self, double[:] x0, Size nSteps=200, Size nQuench=200, double T0=5, double width=1.0) except +:
        assert x0.size == self.nDim, "The starting position given has wrong number of dimensions."
        assert nSteps > 0 and T0 > 0 and width > 0, "The step number, initial temperature, and width must be positive"
        cdef Size d, i
        cdef bint outOfBounds = False
        cdef double energy, energyPropose, temperature
        for d in range(self.nDim):
            self.x[d] = x0[d]
        energy = self.logProbability(self.x,self.gradient,False)
        for i in range(nSteps):
            temperature = T0*(1. - (<double>i)/(nSteps))
            for d in range(self.nDim):
                self.xPropose[d] = normalRand(self.x[d],width*self.scale[d])
                if(self.xPropose[d] > self.upperBoundaries[d] or
                   self.xPropose[d] < self.lowerBoundaries[d]):
                    outOfBounds = True
                    break
            if outOfBounds:
                continue
            energyPropose = self.logProbability(self.xPropose,self.gradient,False)
            if exponentialRand(1.) > (energy - energyPropose)/temperature:
                for d in range(self.nDim):
                    self.x[d] = self.xPropose[d]
                    energy = energyPropose
        for i in range(nQuench):
            for d in range(self.nDim):
                self.xPropose[d] = normalRand(self.x[d],width*self.scale[d]/5.)
                if(self.xPropose[d] > self.upperBoundaries[d] or
                   self.xPropose[d] < self.lowerBoundaries[d]):
                    outOfBounds = True
                    break
            if outOfBounds:
                continue
            energyPropose = self.logProbability(self.xPropose,self.gradient,False)
            if (energyPropose > energy):
                for d in range(self.nDim):
                    self.x[d] = self.xPropose[d]
                    energy = energyPropose
        output = np.zeros(self.nDim)
        for d in range(self.nDim):
            output[d] = self.x[d]
        return output

    cdef void progressBar(self, Size i, Size N, object header) except +:
        f = (10*i)/N
        stdout.write('\r'+header+': <'+f*"="+(10-f)*" "+'> ('+str(i)+" / " + str(N) + ")")
        stdout.flush()
        return 

    cdef void _setMemoryViews_(self) except +:
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
        assert scale.size == nDim, "The scale given has wrong number of dimensions."
        assert logProbability is None or callable(logProbability), "The logProbability is neither callable nor None."
        if upperBoundaries is not None:
            assert upperBoundaries.size == nDim, "The upper boundaries given have the wrong number of dimensions."
        if lowerBoundaries is not None:
            assert lowerBoundaries.size == nDim, "The lower boundaries given have the wrong number of dimensions."
        cdef Size d
        self.nDim = nDim
        self.readyToRun = False
        self.showProgress = True
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
                self._workingMemory_, self.accepted, self.pyLogProbability,
                self.hasBoundaries, self.showProgress)
        return info

    def __setstate__(self,info):
        (self.nDim, self.nSamples, self.burnIn, self.thinning, self.recordStart,
         self.recordStop, self.collectStats, self.readyToRun, self.samplers,
         self._workingMemory_, self.accepted, self.pyLogProbability,
         self.hasBoundaries, self.showProgress) = info
        defaultEngine.setSeed(<unsigned long int>int(os.urandom(4).encode("hex"),16))
        np.random.seed(int(os.urandom(4).encode("hex"),16))
        self._setMemoryViews_()
        self.extraInitialization()
        return

    cdef void extraInitialization(self):
        return
