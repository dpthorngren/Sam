# distutils: language = c++
include "distributions.pyx"
include "griddy.pyx"
import multiprocessing as mp
from scipy.misc import logsumexp
from scipy.linalg import solve_triangular
from sys import stdout
import numpy as np
import inspect
from numpy.linalg import solve, cholesky
import os
cimport numpy as np

# Special function wrappers
cpdef double incBeta(double x, double a, double b):
    return _incBeta(a,b,x)

# Special functions
cpdef double expit(double x) except +:
    return exp(x) / (1 + exp(x))

cpdef double logit(double x) except +:
    return log(x) - log(1-x)

# Helper functions
def getDIC(logLike, samples, full_output=False):
    l = np.array([logLike(i) for i in samples])
    meanLike = np.mean(l)
    nEff = .5*np.var(l)
    if full_output:
        return meanLike, nEff, -2*meanLike + nEff
    return -2*meanLike + nEff

def getAIC(loglike, samples):
    lMax = max([loglike(i) for i in samples])
    return 2*np.shape(samples)[1] - 2*lMax

def getBIC(loglike, samples, nPoints):
    lMax = max([loglike(i) for i in samples])
    return log(nPoints)*np.shape(samples)[1] - 2 * lMax

def gpGaussKernel(x,xPrime,theta):
    return theta[1]*np.exp(-distance(x,xPrime)**2/(2*theta[0]**2))

def gpExpKernel(x,xPrime,theta):
    return theta[1]*np.exp(-distance(x,xPrime)/theta[0])

def distance(x,xPrime):
    x = np.atleast_2d(x.T).T
    xPrime = np.atleast_2d(xPrime.T).T
    return np.sqrt(np.sum((x[:,np.newaxis,:]-xPrime[np.newaxis,:,:])**2,axis=-1))

def gaussianProcess(x, y, theta, xTest=None, kernel=gpExpKernel, kernelChol=None):
    if kernelChol is None:
        K = kernel(x,x,theta) + theta[2]*np.eye(len(x))
        L = cholesky(K)
    else:
        L = kernelChol
    alpha = solve_triangular(L.T,solve_triangular(L,y,lower=True))
    if xTest is not None:
        if x.ndim != 1:
            assert xTest.ndim == 2
            assert x.shape[1] == xTest.shape[1]
        else:
            assert xTest.ndim == 1
        KTest = kernel(x,xTest,theta)
        v = solve_triangular(L,KTest,lower=True)
        predVariance = kernel(xTest,xTest,theta) + np.eye(len(xTest))*theta[2] - np.matmul(v.T,v)
        return np.matmul(KTest.T,alpha), predVariance
    return -.5*np.sum(y*alpha) - np.sum(np.log(np.diag(L)))

def acf(x, length=50):
         if np.ndim(x) == 2:
             return np.array([np.array([1]+[np.corrcoef(x[:-i,j],x[i:,j],0)[0,1] for i in range(1,length)]) for j in range(np.shape(x)[1])]).T
         return np.array([1]+[np.corrcoef(x[:-i],x[i:],0)[0,1] for i in range(1,length)])

cdef class Sam:
    cpdef double logProbability(self, double[:] position, double[:] gradient, bint computeGradient) except +:
        if self.pyLogProbability is None:
            raise NotImplementedError("You haven't defined the log probability,"+
                "but the sampler called it.")
        if self.pyLogProbArgNum == 1:
            if computeGradient:
                raise AttributeError("Gradient information was requested, but the given logProbability function does not provide it.")
            return self.pyLogProbability(np.asarray(position))
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
        cdef double new = inf
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
        mvNormalRand(self.x[dStart:dStop],proposeChol,self.xPropose[dStart:dStop],True)
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
        assert threads > 0, "Threads must be > 0."
        x0 = np.atleast_1d(x0)
        assert (x0.shape == (self.nDim,) or x0.shape == (threads,self.nDim)), "The initial guess must have shape [nDim], or [threads, nDim]."
        self.initialPosition = x0.copy()
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
        # Default to metropolis
        if self.samplers.size() == 0:
            self.addMetropolis(0,self.nDim)
        if threads > 1:
            if x0.ndim == 1:
                x0 = np.array([x0]*threads)
            p = mp.Pool(threads)
            self.samples, self.accepted = zip(*p.map(self,list(x0)))
            self.samples = np.array(self.samples)
            self.accepted = np.array(self.accepted)
            p.terminate()
            self.results = np.reshape(self.samples,(threads*self.nSamples,self.nDim))
            return self.samples
        else:
            self(x0)
            self.results = self.samples
            return self.samples

    def __call__(self, double[:] x0):
        cdef Size i, j, d
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
            self.progressBar(self.nSamples,self.nSamples,"Sampling")
            print ""
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

    cpdef object summary(self, returnString = False) except +:
        assert self.nSamples > 0,"Cannot report statistics without having run the sampler!"
        acceptance = self.getAcceptance()
        if len(acceptance.shape) > 1:
            acceptance = np.mean(acceptance,axis=0)
        means = np.mean(self.results,axis=0)
        stds = np.std(self.results,axis=0)
        percents = np.percentile(self.results,(16,50,84),axis=0)
        output = ("{:<4}"+" {:>6}"+" |"+2*" {:>10}"+" |"+3*" {:>10}").format("Dim.","Accept","Mean","Std.","16%","50%","84%")
        for i in range(self.nDim):
            output += '\n' + ("{:<4}"+" {:>6.1%}"+" |"+2*" {:>10.4g}"+" |"+3*" {:>10.4g}").format(i,acceptance[i],means[i],stds[i],percents[0,i],percents[1,i],percents[2,i])
        output += '\n'
        if returnString:
            return output
        print output
        return


    def getACF(self, length=50):
        assert self.trials > 0, "The number of trials must be greater than zero to compute the autocorrelation function."
        if self.samples.ndim == 2:
            return acf(self.samples)
        if self.samples.ndim == 3:
            return np.array([acf(self.samples[i,:,i]) for i in range(self.samples.shape[0])])

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
            try:
                outputView[d] = (estimate-self.gradient[d])/(estimate+self.gradient[d])
            except:
                outputView[d] = nan
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

    def __init__(self, logProbability, scale, lowerBounds=None, upperBounds=None):
        scale = np.atleast_1d(scale)
        self.nDim = scale.size
        assert logProbability is None or callable(logProbability), "The logProbability is neither callable nor None."
        if upperBounds is not None:
            upperBounds = np.atleast_1d(upperBounds)
            assert upperBounds.size == self.nDim, "The upper boundaries given have the wrong number of dimensions."
        if lowerBounds is not None:
            lowerBounds = np.atleast_1d(lowerBounds)
            assert lowerBounds.size == self.nDim, "The lower boundaries given have the wrong number of dimensions."
        cdef Size d
        self.readyToRun = False
        self.showProgress = True
        self._workingMemory_ = np.nan * np.ones(7*self.nDim,dtype=np.double)
        self.nSamples = 0
        self.accepted = np.zeros(self.nDim,dtype=np.intc)
        self.trials = 0
        self._setMemoryViews_()
        self.pyLogProbability = logProbability
        if self.pyLogProbability is not None:
            self.pyLogProbArgNum = len(inspect.getargspec(self.pyLogProbability).args)
            assert ((self.pyLogProbArgNum == 1) or (self.pyLogProbArgNum == 3)), "The logProbability function must take either one or three arguments."
        else:
            self.pyLogProbArgNum = -1
        for d in range(self.nDim):
            self.scale[d] = scale[d]
        self.hasBoundaries = False
        if upperBounds is not None:
            self.hasBoundaries = True
            for d in range(self.nDim):
                self.upperBoundaries[d] = upperBounds[d]
        else:
            for d in range(self.nDim):
                self.upperBoundaries[d] = inf
        if lowerBounds is not None:
            self.hasBoundaries = True
            for d in range(self.nDim):
                self.lowerBoundaries[d] = lowerBounds[d]
        else:
            for d in range(self.nDim):
                self.lowerBoundaries[d] = -inf
        self.extraInitialization()
        return

    def save(self,filename):
        if self.collectStats:
            stats = self.getStats()
        else:
            stats = np.nan
        if self.trials > 0:
            accept =  self.accepted.astype(np.double)/self.trials
        else:
            accept = np.nan
        try:
            logProbSource = inspect.getsource(self.pyLogProbability)
        except:
            logProbSource = "Source code not available."
        np.savez_compressed(
            filename, nDim=self.nDim, nSamples=self.nSamples,
            thinning=self.thinning, scale=np.asarray(self.scale), upperBounds=self.upperBoundaries,
            lowerBounds=self.lowerBoundaries, initialPosition=self.initialPosition,
            samples = self.samples, acceptance = accept, logProbSource = logProbSource, stats=stats)

    def __getstate__(self):
        info = (self.nDim, self.nSamples, self.burnIn, self.thinning, self.recordStart,
                self.recordStop, self.collectStats, self.readyToRun, self.samplers,
                self._workingMemory_, self.accepted, self.pyLogProbability, self.pyLogProbArgNum,
                self.hasBoundaries, self.showProgress)
        return info

    def __setstate__(self,info):
        (self.nDim, self.nSamples, self.burnIn, self.thinning, self.recordStart,
         self.recordStop, self.collectStats, self.readyToRun, self.samplers,
         self._workingMemory_, self.accepted, self.pyLogProbability, self.pyLogProbArgNum,
         self.hasBoundaries, self.showProgress) = info
        defaultEngine.setSeed(<unsigned long int>int(os.urandom(4).encode("hex"),16))
        np.random.seed(int(os.urandom(4).encode("hex"),16))
        self._setMemoryViews_()
        self.extraInitialization()
        return

    cdef void extraInitialization(self):
        return
