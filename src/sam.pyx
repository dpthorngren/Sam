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
cpdef double incBeta(double x, double a, double b) except -1.:
    return _incBeta(a,b,x)


# Special functions
cpdef double expit(double x) except -1.:
    return exp(x) / (1 + exp(x))


cpdef double logit(double x) except? -1.:
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
    '''A class for sampling from probability distributions.

    Example:
        import numpy as np
        import sam as s
        logProb = lambda x: s.normalLogPDF(x,3,.5)
        f = s.Sam(logProb, np.array([.1]))
        samples = f.run(10000,np.array([0.]))
        print np.mean(samples), np.std(samples)
    '''

    cpdef double logProbability(self, double[:] position, double[:] gradient, bint computeGradient) except? 999.:
        '''Computes the log probability and gradient for the given parameters.

        Args:
            position: an array-like indicating where to evaluate the probability.
            gradient: an array-like where the gradient will be written to.
            computeGradient: A boolean indicating whether a gradient
                is needed.  Used to avoid unnecessary computation.

        Returns:
            The natural log of the probability function, evaluated at the
            given point.  If a gradient was requested, it will have been
            written to the gradient argument.
        '''
        if self.pyLogProbability is None:
            raise NotImplementedError("You haven't defined the log probability,"+
                "but the sampler called it.")
        if self.pyLogProbArgNum == 1:
            if computeGradient:
                raise AttributeError("Gradient information was requested, but the given logProbability function does not provide it.")
            return self.pyLogProbability(np.asarray(position))
        return self.pyLogProbability(np.asarray(position),np.asarray(gradient),computeGradient)

    cdef int sample(self) except -1:
        '''Conducts a single sampling step in the metropolis algorithm.

        The actual act of sampling is determined by which samplers
        have been set -- these are stored in self.samplers.
        The current position is read from self.x, and the arrays
        self.xPropose, self.momentum, and self.gradient may be written
        to during the function call.  self.accepted will be updated
        according to what occurs during the call.
        '''
        cdef size_t s
        cdef Size m
        cdef double logP0 = nan
        for s in range(self.samplers.size()):
            if self.samplers[s].samplerType == 0:
                logP0 = self.metropolisStep(
                    self.samplers[s].dStart,
                    self.samplers[s].dStop, logP0)
            elif self.samplers[s].samplerType == 1:
                logP0 = self.hmcStep(
                    self.samplers[s].idata[0],
                    self.samplers[s].ddata[0],
                    self.samplers[s].dStart,
                    self.samplers[s].dStop, logP0)
            elif self.samplers[s].samplerType == 2:
                m = self.samplers[s].dStop - self.samplers[s].dStart
                logP = self.metropolisCorrStep(
                    self.samplers[s].dStart,
                    self.samplers[s].dStop,
                    np.asarray(self.samplers[s].ddata).reshape((m,m)),logP0)
            elif self.samplers[s].samplerType == 3:
                logP = self.adaptiveStep(
                    self.samplers[s].dStart,
                    self.samplers[s].dStop,
                    &self.samplers[s].ddata,
                    &self.samplers[s].idata, logP0)
        return 0

    cdef double hmcStep(self,Size nSteps, double stepSize, Size dStart, Size dStop, double logP0=nan) except 999.:
        '''Conducts a single Hamiltonian Monte Carlo trajectory.

        The operation is conducted on the internal variables of the object,
        writing the result to self.x and using self.xPropose, x.momentum,
        and x.gradient as scratch space.  self.accepted is updated
        according to whether the trajectory was accepted.

        Args:
            nSteps: The number of steps in the trajectory
            stepSize: Multiplied by self.scale to scale the process.
            dStart: The index of the first parameter to be included.
            dStop: The index of the last parameter to be included, plus one.
            logP0: If the logProbability is known at the starting position,
                providing this will save a small amount of time by not
                recomputing it.

        Returns:
            The log probability at the final position of the trajectory.  This
            is provided to help reduce unnecessary computation; the main
            purpose of this function is actually to advance self.x forward
            one MCMC step.
        '''
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

    cdef int bouncingMove(self, double stepSize, Size dStart, Size dStop) except -1:
        '''Attempts to move self.xPropose, bouncing off of any boundaries.

        The update rule is: x_p[n+1] = x_p[n] + p[n] * stepSize * scale,
        where x_p is the proposed x position (self.xProposed, p is the momentum
        (self.momentum), and scale is the global scaling vector (self.scale).
        If a boundary is encountered, the particle bounces of the boundary
        normal vector and continues moving.

        Args:
            stepSize: Scales the distance traveled (see above).  In classical
                mechanics, this is the time-step.
            dStart: The index of the first parameter to be included.
            dStop: The index of the last parameter to be included, plus one.
        '''
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
        return 0

    cdef double adaptiveStep(self, Size dStart, Size dStop, vector[double]* ddata, vector[int]* idata, double logP0=nan) except 999.:
        # TODO: Documentation
        cdef Size i, j
        cdef Size n = dStop - dStart
        cdef double[:,:] chol
        # Cast the state vector as the mean, covariance, cholesky, and epsilon
        cdef double eps = ddata[0][0]
        cdef double[:] mu = <double[:n]> &ddata[0][1]
        cdef double[:,:] covar = <double[:n,:n]> &ddata[0][1+n]
        cdef double[:,:] covChol = <double[:n,:n]> &ddata[0][1+n+n*n]
        cdef int* t = &idata[0][0]
        self.onlineCovar(mu,covar,self.x,t[0],eps)
        t[0] += 1
        # TODO: Add update frequency option for adaptive step
        if (t[0] >= idata[0][1]) and (t[0]%idata[0][2] == 0):
            chol = cholesky(np.asarray(covar))
            for i in range(covar.shape[0]):
                for j in range(i):
                    covChol[i,j] = chol[i,j]
        return self.metropolisCorrStep(dStart, dStop, covChol,logP0)

    cdef double metropolisStep(self, Size dStart, Size dStop, double logP0=nan) except 999.:
        '''Conducts a single Metropolis-Hastings step.

        The proposal distribution is a normal distribution centered at self.x,
        with diagonal covariance where self.scale is the diagonal components.
        The operation is conducted on the internal variables of the object,
        writing the result to self.x and using self.xPropose, x.momentum,
        and x.gradient as scratch space.  self.accepted is updated
        according to whether the trajectory was accepted.

        Args:
            dStart: The index of the first parameter to be included.
            dStop: The index of the last parameter to be included, plus one.
            logP0: If the logProbability is known at the starting position,
                providing this will save a small amount of time by not
                recomputing it.

        Returns:
            The log probability at the final position of the trajectory.  This
            is provided to help reduce unnecessary computation; the main
            purpose of this function is actually to advance self.x forward
            one MCMC step.
        '''
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

    cdef double metropolisCorrStep(self, Size dStart, Size dStop, double[:,:] proposeChol, double logP0=nan) except 999.:
        '''Conducts a single Metropolis-Hastings step for a general covariance.

        The proposal distribution is a normal distribution centered at self.x,
        with the covariance given by the argument proposeChol, which is the
        cholesky of the proposal covariance matrix.  The operation is
        conducted on the internal variables of the object, writing the result
        to self.x and using self.xPropose, x.momentum, and x.gradient as
        scratch space.  self.accepted is updated according to whether the
        trajectory was accepted.

        Args:
            dStart: The index of the first parameter to be included.
            dStop: The index of the last parameter to be included, plus one.
            proposeChol: A [N x N] array-like which is the cholesky of the
                desired proposal covariance, and N is dStop-dStart-1
            logP0: If the logProbability is known at the starting position,
                providing this will save a small amount of time by not
                recomputing it.

        Returns:
            The log probability at the final position of the trajectory.  This
            is provided to help reduce unnecessary computation; the main
            purpose of this function is actually to advance self.x forward
            one MCMC step.
        '''
        cdef Size d
        cdef double logP1
        mvNormalRand(self.x[dStart:dStop],proposeChol,self.xPropose[dStart:dStop],True)
        for d in range(0,self.nDim):
            if d >= dStart and d < dStop:
                if self.hasBoundaries and (self.xPropose[d] > self.upperBoundaries[d] or
                   self.xPropose[d] < self.lowerBoundaries[d]):
                    # TODO: Smart reflection, rather than rejection
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

    cdef double[:] regressionStep(self, double[:,:] x1, double[:] y1, double[:] output=None) except *:
        '''Draws from a Bayesian linear regression with normal errors on x,y.

        This is a sampler, and so the return is not the maximum likelihood
        coefficients, but rather a sample drawn from the probability
        distribution about the maximum (in this case, the MLE is also the
        posterior mean).  Priors are assumed to be flat, except for the
        residual standard deviation, which is p ~ 1/sigma^2.

        Args:
            x1: The design matrix, viz. columns of predictor variables stacked
                horizontally into a matrix.  Should be [N x P] where N is the
                number of data points and P is the number of predictors.
            y1: The dependent data to be fitted against.  Should be length N.
            output: An array-like to write the resulting samples to.  Should be
                length P+1, one for each predictor plus the standard deviation.

        Returns:
            A sample from the posterior distribution for the coefficients of
            the predictors as well as the standard deviation.  If output was
            given, this will simply be a reference to it..
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


    cpdef object addMetropolis(self, covariance=None, Size dStart=0, Size dStop=-1):
        '''Adds a metropolis sampler with a non-diagonal covariance.

        This sampler sets up a Metropolis-Hastings sampler to be used during
        the sampling procedure.  The proposal distribution is a normal
        distribution centered at self.x, with a covariance supplied by the
        user.  If no covariance is supplied, default is a matrix
        with self.scale (set during initialization) as the diagonal (this is
        slightly faster to produce random numbers for).

        Args:
            covariance: The covariance matrix to be used.  Should be [M x M],
                where M=dStop-dStart.  May be None to use a diagonal matrix
                with self.scale as the diagonals.
            dStart: The index of the first parameter to be included. Default
                is zero.
            dStop: The index of the last parameter to be included, plus one.
                Default is the last index + 1.
        '''
        if dStop < 0:
            dStop = self.nDim
        assert dStart >= 0 and dStart < self.nDim, "The start parameter must be between 0 and nDim - 1 (inclusive)."
        assert dStop > 0 and dStop <= self.nDim, "The stop parameter must be between 1 and nDim (inclusive)."
        cdef SamplerData samp
        if covariance is not None:
            assert covariance.shape[0] == covariance.shape[1] == dStop-dStart
            samp.samplerType = 2
            covChol = cholesky(covariance)
            samp.ddata = covChol.flatten()
        else:
            samp.samplerType = 0
        samp.dStart = dStart
        samp.dStop = dStop
        self.samplers.push_back(samp)
        return


    cpdef object addAdaptiveMetropolis(self, covariance=None, int adaptAfter=-1, int refreshPeriod=100, double eps=1e-9, Size dStart=0, Size dStop=-1):
        '''Adds an Adaptive Metropolis sampler to the sampling procedure.

        This sampler is the Adaptive Metropolis (AM) algorithm presented in
        Haario et al. (2001).  The algorithm initially samples with a given
        proposal covariance, but after a number of steps, uses the covariance
        of the samples to estimate the optimal proposal covariance.  Each time
        the propsal is updated, the cholesky (order n^3) must be computed, so
        it may be advisable not to set the refresh period too low if n is
        large.  Note that the estimated covariance is updated every time the
        sampler is called, but that this is not propagated to the sampler
        until the refresh occurs.

        Args:
            covariance: The initial proposal covariance to sample with.
                Should be [M x M], where M = nStop - nStart.
            adaptAfter: How many times the sampler must be called (and thereby
                collect samples) before the adapted covariance is used.  Must
                be larger than the number of dimensions being adapted to.  The
                default (triggered by any negative number) is three times that.
            refreshPeriod: How many times the sampler is called between the
                cholesky of the covariance being updated (the expensive part).
            eps: The epsilon parameter in Haario et al. (2001).  It needs to be
                small but nonzero for the theory to work, but in practice
                seems to work at zero as well.  Default is 1e-9, and probably
                will not need to be changed by the user unless the covariance
                is on that scale or less.
            dStart: The index of the first parameter to be included. Default
                is zero.
            dStop: The index of the last parameter to be included, plus one.
                Default is the last index + 1.
        '''
        if dStop < 0:
            dStop = self.nDim
        assert dStart >= 0 and dStart < self.nDim, "The start parameter must be between 0 and nDim - 1 (inclusive)."
        assert dStop > 0 and dStop <= self.nDim, "The stop parameter must be between 1 and nDim (inclusive)."
        assert refreshPeriod > 0
        if adaptAfter < 0:
            adaptAfter = 3*(dStop-dStart)
        assert dStop-dStart < adaptAfter
        if covariance is None:
            covariance = np.diag(np.asarray(self.scale)[dStart:dStop])
        assert covariance.shape[0] == covariance.shape[1] == dStop-dStart, "Misshapen covariance."
        cdef SamplerData samp
        covChol = cholesky(covariance)
        samp.samplerType = 3
        samp.dStart = dStart
        samp.dStop = dStop
        samp.idata.push_back(0)
        samp.idata.push_back(adaptAfter)
        samp.idata.push_back(refreshPeriod)
        samp.ddata = np.concatenate([[eps],np.zeros(covariance.shape[0]),
                                          np.zeros(covariance.shape[0]**2),
                                          covChol.flatten()])
        self.samplers.push_back(samp)
        return

    cpdef object addHMC(self, Size nSteps, double stepSize, Size dStart=0, Size dStop=-1):
        '''Adds a Hamiltonian Monte Carlo sampler to the sampling procedure.

        This sampler sets up an HMC sampler to be used during the sampling
        procedure.  The proposal distribution is a normal distribution
        centered at self.x, with diagonal covariance where the scale (set
        when the sampler object was initialized) is the diagonal components.

        Args:
            nSteps: The number of steps in the trajectory
            stepSize: Multiplied by self.scale to scale the process.
            dStart: The index of the first parameter to be included. Default
                is zero.
            dStop: The index of the last parameter to be included, plus one.
                Default is the last index + 1.
        '''
        if dStop < 0:
            dStop = self.nDim
        assert dStart >= 0 and dStart < self.nDim, "The start parameter must be between 0 and nDim - 1 (inclusive)."
        assert dStop > 0 and dStop <= self.nDim, "The stop parameter must be between 1 and nDim (inclusive)."
        assert nSteps > 0 and stepSize > 0, "The step size and the number of steps must be greater than zero."
        cdef SamplerData samp
        samp.samplerType = 1
        samp.idata.push_back(nSteps)
        samp.ddata.push_back(stepSize)
        samp.dStart = dStart
        samp.dStop = dStop
        self.samplers.push_back(samp)
        return

    cpdef object getSampler(self, unsigned int i=0):
        assert i < self.samplers.size()
        return self.samplers[i]

    cpdef object getProposalCov(self, unsigned int i=0):
        assert i < self.samplers.size()
        if self.samplers[i].samplerType == 0:
            return np.diag(self.scale)
        elif self.samplers[i].samplerType == 1:
            return np.diag(1./np.asarray(self.scale))
        elif self.samplers[i].samplerType == 2:
            n = self.samplers[i].dStop - self.samplers[i].dStart
            chol = np.asarray(self.samplers[i].ddata).reshape(n,n)
            return np.matmul(chol,chol.T)
        elif self.samplers[i].samplerType == 3:
            n = self.samplers[i].dStop - self.samplers[i].dStart
            return np.asarray(self.samplers[i].ddata[1+n:1+n+n**2]).reshape(n,n)


    cpdef object printSamplers(self):
        '''Prints the list of any/all sampling systems set up so far.

        This refers to samplers added using e.g. addMetropolis() functions. See
        the overall class documentation for more information on these.  The
        type of function and any parameters given (unique to each sampler
        type) are printed.
        '''
        cdef size_t s
        if self.samplers.size() == 0:
            print "No samplers defined."
        for s in range(self.samplers.size()):
            if self.samplers[s].samplerType == 0:
                print s, "Diagonal Metropolis ("+str(self.samplers[s].dStart)+":"+str(self.samplers[s].dStop)+")"
                print '\tdiag(C) =', np.asarray(self.scale)
            elif self.samplers[s].samplerType == 1:
                print s, "HMC ("+str(self.samplers[s].dStart)+":"+str(self.samplers[s].dStop)+"), ",\
                    self.samplers[s].idata[0], "steps with size", self.samplers[s].ddata[0]
            elif self.samplers[s].samplerType == 2:
                print s, "Metropolis ("+str(self.samplers[s].dStart)+":"+str(self.samplers[s].dStop)+"), Cov ="
                n = self.samplers[s].dStop - self.samplers[s].dStop
                print np.asarray(self.samplers[s].ddata).reshape(n,n)
            elif self.samplers[s].samplerType == 3:
                print s, "Adaptive Metropolis ("+str(self.samplers[s].dStart)+":"+str(self.samplers[s].dStop)+"),",\
                    "Start adapting after", self.samplers[s].idata[1], \
                    "steps, updating every", self.samplers[s].idata[2], "steps."
        return

    cpdef object clearSamplers(self):
        '''Clears the list of samplers.'''
        self.samplers.clear()

    cdef int record(self,Size i) except -1:
        '''Internal function that records the current position to self.samples.

        Only parameters with indices between recordStart and recordStop
        (set in the run() function and stored as a class member) are
        recorded.

        Args:
            i: The index to store the current position at.
        '''
        cdef Size d
        for d in range(self.recordStart,self.recordStop):
            self.sampleView[i,d-self.recordStart] = self.x[d]
        return 0

    cdef int recordStats(self) except -1:
        '''Internal function that accumulates statistics about the samples.

        This function is called each time a sample needs to be
        accumulated into the statistics tracker.  If self.collectStats is False
        (set in the run() function), this function will never be called.
        '''
        cdef Size d
        for d in range(self.nDim):
            self.sampleStats[d](self.x[d])
        return 0

    cpdef object run(self, Size nSamples, object x0, Size burnIn=0, Size thinning=0, Size recordStart=0, Size recordStop=-1, bint collectStats=False, Size threads=1, bint showProgress=True):
        '''Begin sampling the parameters from the given logProbability dist.

        Args:
            nSamples: The desired number of samples to record per thread.
            burnIn:  The number of MCMC steps to take before recording begins.
            thinning: The number of MCMC steps to take between recordings and
                burn-in steps.  This directly multiplies the amount of work
                the sampler needs to do.
            recordStart: The index of the first parameter to be recorded.
                Default is 0.
            recordStop: The index of the last parameter to be recorded, plus one.
                Default is the number of parameters (all are recorded).
            collectStats: Whether the sampler should collect running statistics
                as it runs.  This is probably only desirable if not all
                parameters are being recorded.  Default is False
            threads: The number of computational threads to run in.  If this is
                greater than 1, the multiprocessing library is used to run
                this many fully independent samplers in parallel.
            showProgress: Whether to print a progress bar as the sampler runs.

        Returns:
            The parameter samples, of the shape [N x M X T], where N is the
            number of parameters of target distribution, M is the number of
            samples requested (nSamples), and T is the number of threads
            used.  If threads is 1 the last dimension is omitted.  This data
            is also stored internally and can be accessed later using other
            functions.
        '''
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
            self.addMetropolis()
        if threads > 1:
            if x0.ndim == 1:
                x0 = np.array([x0]*threads)
            p = mp.Pool(threads)
            try:
                self.samples, self.accepted = zip(*p.map_async(self,list(x0)).get(1000000000))
                p.terminate()
                self.samples = np.array(self.samples)
                self.accepted = np.array(self.accepted)
                self.results = np.reshape(self.samples,(threads*self.nSamples,self.nDim))
                return self.samples
            except KeyboardInterrupt:
                p.terminate()
                self.readyToRun = False
            return None
        else:
            self(x0)
            self.results = self.samples
            return self.samples

    def __call__(self, double[:] x0):
        '''Internal function used to run the sampler.  Not for user use!

        This function is an internal way to sampler, which assumes certain
        other internal parameters have already been setup.  Users should call
        the run() function.  This function only exists because it makes running
        the sampler in parallel simpler to code.

        Args:
            x0: The initial position for the run.
        '''
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

    cpdef object getStats(self):
        '''Returns running-average and standard deviation statistics.

        Note that this is only applicable if collectStats was enabled during
        the sampling procedure (default is off).  You will need to compute the
        values directly from the sample otherwise.  This is intended as a tool
        for tracking nuisance parameters for which it was not worth recording
        the samples from -- see recordStart and recordStop in the self.run
        arguments.

        Returns:
            An array of means and an array of standard deviations of the
            parameter samples.
        '''
        assert self.collectStats, "Running statistics collection is turned off."
        assert self.trials > 0, "Cannot report statistics without having run the sampler!"
        assert self.sampleStats.size() > 0, "Cannot report statistics without having run the sampler!"
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
        '''Calculates the acceptance rate for each dimension.

        This includes burn-in and thinning samples.  Throws an
        assertion exception if the sampler has not yet been run.

        Returns:
            The fraction of parameter samples which were accepted, with the
            shape [M X T], number of parameters of target distribution and
            T is the number of threads used.  If threads is 1 the second
            dimension is omitted.
        '''
        assert self.trials > 0, "The number of trials must be greater than zero to compute the acceptance rate."
        return self.accepted.astype(np.double)/self.trials

    cpdef object summary(self, paramIndices=None, returnString=False):
        '''Prints/returns some summary statistics of the previous sampling run.

        Statistics are the parameter index, the acceptance rate, mean, and
        standard deviation, as well as teh 16th, 50th, and 84th percentiles.

        Args:
            paramIndices: The indices of the parameters to be described.  If
            set to None, then all parameters will be described.

        Returns:
            The summary message as a string if returnString is True, otherwise
            returns None.

        '''
        assert self.nSamples > 0,"Cannot report statistics without having run the sampler!"
        acceptance = self.getAcceptance()
        if len(acceptance.shape) > 1:
            acceptance = np.mean(acceptance,axis=0)
        if paramIndices is None:
            paramIndices = range(self.nDim)
        means = np.mean(self.results,axis=0)
        stds = np.std(self.results,axis=0)
        percents = np.percentile(self.results,(16,50,84),axis=0)
        output = ("{:<4}"+" {:>6}"+" |"+2*" {:>10}"+" |"+3*" {:>10}").format("Dim.","Accept","Mean","Std.","16%","50%","84%")
        for i in paramIndices:
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

    cpdef object testGradient(self, double[:] x0, double eps=1e-5):
        '''Compares gradients from logProbability and finite difference method

        This function computes a gradient estimate using the finite difference
        method and compares it to the gradient computed by the logProbability
        function (specified at initialization).

        Args:
            x0: The central location to compute the gradient at.
            eps: scales the finite difference method.  Secondary samples are
                taken in the direction of each parameter a distance scale*eps
                away.

        Returns:
            The relative difference between the logProbability gradient L and
            the finite difference gradient F: (L-F)/L+F)
        '''
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

    cpdef object gradientDescent(self, double[:] x0, double step=.1, double eps=1e-10):
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

    cpdef object simulatedAnnealing(self, double[:] x0, Size nSteps=200, Size nQuench=200, double T0=5, double width=1.0):
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

    cdef int progressBar(self, Size i, Size N, object header) except -1:
        '''Displays or updates a simple ASCII progress bar.

        Args:
            i: the current iteration, should <= N.
            N: the total number of iterations to be done.
            header: a string to display before the progress bar.

        Example::
            self.progressBar(55,100,"Something")
            # Restarts the line and prints:
            Something: <=====     > (55/100)
        '''
        f = (10*i)/N
        stdout.write('\r'+header+': <'+f*"="+(10-f)*" "+'> ('+str(i)+" / " + str(N) + ")          ")
        stdout.flush()
        return 0

    cdef int onlineCovar(self, double[:] mu, double[:,:] covar, double[:] x, int t, double eps=1e-9) except -1:
        # TODO: Documentation
        cdef Size i, j
        cdef Size n = mu.shape[0]
        cdef double scalingParam = 5.76/n
        # TODO: make more efficient?
        cdef double[:] muOld = mu.copy()
        for i in range(n):
            mu[i] = (t*mu[i]+x[i])/(t+1)
        if t > 0:
            for i in range(n):
                for j in range(n):
                    covar[i,j] = (t-1)*covar[i,j] + scalingParam*(t*muOld[i]*muOld[j] - (t+1)*mu[i]*mu[j] + x[i]*x[j])
                    if i == j:
                        covar[i,j] += eps
                    covar[i,j] /= t
        return 0

    cdef int _setMemoryViews_(self) except -1:
        '''Sets up the memoryviews of the working memory in the sampler.

        This function ensures that the various memoryviews used by the class
        point towards the right parts of the working memory array.  It
        should never need to be called directly by the user.
        '''
        self.x = self._workingMemory_[0:self.nDim]
        self.xPropose = self._workingMemory_[self.nDim:2*self.nDim]
        self.momentum = self._workingMemory_[2*self.nDim:3*self.nDim]
        self.gradient = self._workingMemory_[3*self.nDim:4*self.nDim]
        self.scale = self._workingMemory_[4*self.nDim:5*self.nDim]
        self.upperBoundaries = self._workingMemory_[5*self.nDim:6*self.nDim]
        self.lowerBoundaries = self._workingMemory_[6*self.nDim:7*self.nDim]
        self.acceptedView = self.accepted
        return 0

    def __init__(self, logProbability, scale, lowerBounds=None, upperBounds=None):
        '''Instantiates the sampler class and sets the logProbability function.

        Args:
            logProbability: A function which takes one or three arguments and
                returns the natural log of the probability evaluated at the
                position given by the first argument.  If three arguments are
                allowed, the second argument provides an array to write the
                gradient to, and the third argument is a bool indicating
                whether a gradient is required for that call to the function.
            scale: An array whose length  is the number of parameters in the
                target distribution.
            lowerBounds: An array whose length is the number of parameters
                which defines lower boundaries below which the parameters
                will not be sampled.  The sampler is optimized so that these
                boundaries are enforced efficiently, and so will not decrease
                the acceptance rate.  If None, no boundaries are enforced.
            upperBounds: Same as lowerBounds, but defines the upper boundaries

        Returns:
            An instantiated object of the class.
        '''
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
        '''Saves the results of sampling and other information to an npz file.

        To be exact, it saves the current filename, the number of parameters,
        the number of samples collected, the thinning amount, the number of
        burn-in samples, the scale vector, the bounds vectors, the initial
        position vector (or matrix), the samples themselves, the acceptance
        rate, accumulated statistics, and the logProbability source code.  If
        some of those are not available, placeholders are saved instead.

        Args:
            filename: The name of the file to save the information into.  The
            suffix '.npz' will automatically be added.

        Returns:
            None
        '''
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
            filename, nDim=self.nDim, nSamples=self.nSamples, burnIn = self.burnIn,
            thinning=self.thinning, scale=np.asarray(self.scale), upperBounds=self.upperBoundaries,
            lowerBounds=self.lowerBoundaries, initialPosition=self.initialPosition,
            samples = self.samples, acceptance = accept, logProbSource = logProbSource, stats=stats)

    def __getstate__(self):
        '''Prepares internal memory for pickling.

        This is important for compatibility with the multiprocessing library
        used for parallelism.

        Returns:
            A pickleable tuple of the internal variables.
        '''
        info = (self.nDim, self.nSamples, self.burnIn, self.thinning, self.recordStart,
                self.recordStop, self.collectStats, self.readyToRun, self.samplers,
                self._workingMemory_, self.accepted, self.pyLogProbability, self.pyLogProbArgNum,
                self.hasBoundaries, self.showProgress)
        return info

    def __setstate__(self,info):
        '''Restores internal memory from pickle info and resets the RNG.

        This is important for compatibility with the multiprocessing library
        used for parallelism.

        Args:
            info: The information from a pickle.  This almost certainly comes
            from the __getstate__ function.

        Returns:
            None
        '''
        (self.nDim, self.nSamples, self.burnIn, self.thinning, self.recordStart,
         self.recordStop, self.collectStats, self.readyToRun, self.samplers,
         self._workingMemory_, self.accepted, self.pyLogProbability, self.pyLogProbArgNum,
         self.hasBoundaries, self.showProgress) = info
        defaultEngine.setSeed(<unsigned long int>int(os.urandom(4).encode("hex"),16))
        np.random.seed(int(os.urandom(4).encode("hex"),16))
        self._setMemoryViews_()
        self.extraInitialization()
        return

    cdef int extraInitialization(self) except -1:
        '''This function can be overloaded to do extra initialization.

        By default it does nothing.  It is provided in case the user wishes
        to subclass Sam and needs a function which is always called when the
        object is instantiated or copied for multithreading.

        It should return 0 to indicate no exception occurred.  This is the only
        purpose of its return value
        '''
        return 0
