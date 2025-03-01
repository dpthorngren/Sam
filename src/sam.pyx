# distutils: language = c++
import multiprocessing as mp
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
import time
import sys
import numpy as np
import inspect
from numpy.linalg import solve, cholesky
import os
cimport numpy as np

include "distributions.pyx"
include "gaussianProcess.pyx"
include "griddy.pyx"


# Special function wrappers
cpdef double incBeta(double x, double a, double b) except -1.:
    return _incBeta(a,b,x)


# Special functions
cpdef double expit(double x) except -1.:
    return exp(x) / (1 + exp(x))


cpdef double logit(double x) except? -1.:
    return log(x) - log(1-x)


# Linear algebra wrappers
cpdef int choleskyInplace(double[:,:] x) except -1:
    if x is None:
        raise ValueError("Input must not be None!")
    if x.shape[0] != x.shape[1]:
        raise ValueError("Must be a square matrix. Got:"+str(x.shape[0])+" x "+str(x.shape[1]))
    cdef int n = x.shape[0]
    cdef int output = 0
    lapack.dpotrf('U',&n,&x[0,0],&n,&output)
    if output != 0:
        raise ValueError("Matrix was not positive definite!")
    return 0


# Helper functions
def getDIC(logLikeSamples):
    '''A simple helper function to compute the DIC given its arguments.

    args:
        logLikeSamples: the log (base e) likelihood of MCMC samples of
            the model of interest.

    Returns:
        The estimated DIC for the model as a floating point number.
    '''
    meanLike = np.mean(logLikeSamples)
    nEff = .5*np.var(logLikeSamples)
    return -2*meanLike + nEff


def getAIC(maxLogLike, nParams):
    '''A simple helper function to compute the AIC given its arguments.

    args:
        maxLogLike: the log (base e) of the maximum likelihood of the model.
        nParams: the number of parameters used in the model

    Returns:
        The estimated AIC for the model as a floating point number.
    '''
    return 2*nParams - 2*maxLogLike


def getBIC(maxLogLike, nParams, nPoints):
    '''A simple helper function to compute the BIC given its arguments.

    args:
        maxLogLike: the log (base e) of the maximum likelihood of the model.
        nParams: the number of parameters used in the model
        nPoints: the number of data points used in evaluating the likelihood

    Returns:
        The estimated BIC for the model as a floating point number.
    '''
    return log(nPoints)*nParams - 2*maxLogLike


def acf(x, length=50):
         if np.ndim(x) == 2:
             return np.array([np.array([1]+[np.corrcoef(x[:-i,j],x[i:,j],0)[0,1] for i in range(1,length)]) for j in range(np.shape(x)[1])]).T
         return np.array([1]+[np.corrcoef(x[:-i],x[i:],0)[0,1] for i in range(1,length)])


def gelmanRubin(x, warn=True):
    x = np.asarray(x, np.double, "C")
    if x.ndim == 2:
        if warn:
            print("Warning: the G.R. diagnostic was not designed for " +\
                "the case where chains are not completely independent.")
        x = np.stack([x[:x.shape[0]//2],x[x.shape[0]//2:]],axis=0)
    elif x.ndim != 3:
        raise ValueError("Input has an invalid shape: must be 3-d.")
    n = np.shape(x)[1]
    W = np.mean(np.var(x,axis=1,ddof=1),axis=0)
    B_n = np.var(np.mean(x,axis=1),axis=0,ddof=1)
    return np.sqrt((1.-1./n) + B_n/W)


cdef class Sam:
    '''A class for sampling from probability distributions.'''

    cpdef double _logProbability_(self, double[:] position, double[:] gradient, bint computeGradient) except? 999.:
        '''Tries to get a surrogate model estimate of the logProbability if
            useSurrogate is set, otherwise just calls logProbability.

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
        cdef Size i
        if self.useSurrogate:
            scaledPosition = np.asarray(position)/np.asarray(self._scale)
            p, pErr = self.surrogate.predict(scaledPosition[None,:])
            p += self.surrogateOffset
            if pErr > self.surrogateTol:
                # Surrogate precision is inadequate, call true function
                p = self.logProbability(position,gradient,False)
                # Update surrogate with this new datapoint
                params = np.asarray(self.surrogate.params).copy()
                newY = np.concatenate([np.asarray(self.surrogate.y)+self.surrogateOffset,[p]])
                self.surrogateOffset = np.median(newY)
                self.surrogate = GaussianProcess(
                    np.vstack([self.surrogate.x,scaledPosition]),
                    newY - self.surrogateOffset,
                    self.surrogate.kernelName)
                self.surrogateSamples += 1
                if self.surrogateSamples > self.surrogateUpdateRate*self.surrogateLastOptimize:
                    self.surrogate.optimizeParams(params)
                    self.surrogateLastOptimize = self.surrogateSamples
                else:
                    self.surrogate.precompute(params)
            if computeGradient:
                self.surrogate._gradient_(scaledPosition,gradient)
            return p
        return self.logProbability(position,gradient,computeGradient)

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
        elif self.pyLogProbArgNum == 2:
            if computeGradient:
                raise AttributeError("Gradient information was requested, but the given logProbability function does not provide it.")
            if self.userParametersView.shape[0] == 0:
                raise AttributeError("User parameters were requested but have not yet been set with setUserParameters(...).")
            return self.pyLogProbability(np.asarray(position), np.asarray(self.userParametersView))
        elif self.pyLogProbArgNum == 4:
            if self.userParametersView.shape[0] == 0:
                raise AttributeError("User parameters were requested but have not yet been set with setUserParameters(...).")
            return self.pyLogProbability(np.asarray(position), np.asarray(self.userParametersView), np.asarray(gradient), computeGradient)
        return self.pyLogProbability(np.asarray(position), np.asarray(gradient), computeGradient)

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
        for s in range(self.samplers.size()):
            if self.samplers[s].samplerType == 0:
                self.lastLogProb = self.metropolisStep(
                    self.samplers[s].dStart,
                    self.samplers[s].dStop,
                    self.lastLogProb)
            elif self.samplers[s].samplerType == 1:
                self.lastLogProb = self.hmcStep(
                    self.samplers[s].idata[0],
                    self.samplers[s].ddata[0],
                    self.samplers[s].dStart,
                    self.samplers[s].dStop,
                    self.lastLogProb)
            elif self.samplers[s].samplerType == 2:
                m = self.samplers[s].dStop - self.samplers[s].dStart
                self.lastLogProb = self.metropolisCorrStep(
                    self.samplers[s].dStart,
                    self.samplers[s].dStop,
                    <double[:m,:m]> &self.samplers[s].ddata[0],
                    self.lastLogProb)
            elif self.samplers[s].samplerType == 3:
                self.lastLogProb = self.adaptiveStep(
                    self.samplers[s].dStart,
                    self.samplers[s].dStop,
                    &self.samplers[s].ddata,
                    &self.samplers[s].idata,
                    self.lastLogProb)
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
        cdef double logP1 = inf
        for d in range(self.nDim):
            self.xPropose[d] = self.x[d]
            if d >= dStart and d < dStop:
                self.momentum[d] = normalRand(0,1./self._scale[d])
            else:
                self.momentum[d] = 0

        # Compute the kinetic energy part of the initial Hamiltonian
        cdef double kinetic = 0
        for d in range(dStart,dStop):
            kinetic += self.momentum[d]*self.momentum[d]*self._scale[d]*self._scale[d] / 2.0

        # Simulate the trajectory
        if isnan(logP0):
            logP0 = self._logProbability_(self.xPropose,self.gradient,True)
        for i in range(nSteps):
            for d in range(dStart,dStop):
                self.momentum[d] += stepSize * self.gradient[d] / 2.0
                self.xPropose[d] += self.momentum[d] * stepSize * self._scale[d] * self._scale[d]
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
            logP1 = self._logProbability_(self.xPropose, self.gradient,True)
            for d in range(dStart,dStop):
                self.momentum[d] += stepSize * self.gradient[d] / 2.0

        # Compute the kinetic energy part of the proposal Hamiltonian
        cdef double kineticPropose = 0
        for d in range(dStart,dStop):
            kineticPropose += self.momentum[d]*self.momentum[d]*self._scale[d]*self._scale[d]/2.0

        # Decide whether to accept the new point
        if isnan(logP1) or isnan(logP0):
            raise ValueError("Got NaN for the log probability!")
        if (exponentialRand(1.) > logP0 - kinetic - logP1 + kineticPropose):
            for d in range(dStart,dStop):
                self.acceptedView[d] += 1
                self.x[d] = self.xPropose[d]
            return logP1
        return logP0

    cdef double adaptiveStep(self, Size dStart, Size dStop, vector[double]* ddata, vector[int]* idata, double logP0=nan) except 999.:
        # TODO: Documentation
        cdef Size i, j
        cdef Size n = dStop - dStart
        cdef double[:,:] chol
        # Cast the state vector as the mean, covariance, cholesky, and epsilon
        cdef double eps = ddata[0][0]
        cdef double scaling = ddata[0][1]
        cdef double[:] mu = <double[:n]> &ddata[0][2]
        cdef double[:,:] covar = <double[:n,:n]> &ddata[0][2+n]
        cdef double[:,:] covChol = <double[:n,:n]> &ddata[0][2+n+n*n]
        cdef int* t = &idata[0][0]
        cdef int adaptAfter = idata[0][1]
        cdef int recordAfter = idata[0][2]
        cdef int refreshPeriod = idata[0][3]
        if t[0] > recordAfter:
            self.onlineCovar(mu,covar,self.x,t[0],scaling,eps)
        # TODO: Add update frequency option for adaptive step
        if (t[0] >= adaptAfter) and ((t[0]-adaptAfter)%refreshPeriod == 0):
            chol = cholesky(np.asarray(covar, np.double, "C"))
            for i in range(covar.shape[0]):
                for j in range(i):
                    covChol[i,j] = chol[i,j]
        t[0] += 1
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
                self.xPropose[d] = normalRand(self.x[d], self._scale[d])
                while self.hasBoundaries:
                    if self.xPropose[d] >= self.upperBoundaries[d]:
                        self.xPropose[d] = 2*self.upperBoundaries[d] - self.xPropose[d]
                        continue
                    if self.xPropose[d] <= self.lowerBoundaries[d]:
                        self.xPropose[d] = 2*self.lowerBoundaries[d] - self.xPropose[d]
                        continue
                    break
            else:
                self.xPropose[d] = self.x[d]
        if isnan(logP0):
            logP0 = self._logProbability_(self.x,self.gradient,False)
        logP1 = self._logProbability_(self.xPropose,self.gradient,False)
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
                desired proposal covariance, and N is dStop-dStart
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
            logP0 = self._logProbability_(self.x,self.gradient,False)
        logP1 = self._logProbability_(self.xPropose,self.gradient,False)
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
            assert covariance.shape[0] == covariance.shape[1] == dStop-dStart, "Misshapen covariance."
            samp.samplerType = 2
            covChol = cholesky(covariance)
            samp.ddata = covChol.flatten()
        else:
            samp.samplerType = 0
        samp.dStart = dStart
        samp.dStop = dStop
        self.samplers.push_back(samp)
        return


    cpdef object addAdaptiveMetropolis(self, covariance=None, int adaptAfter=-1, int recordAfter=-1, int refreshPeriod=100, double scaling=-1, double eps=1e-9, Size dStart=0, Size dStop=-1):
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
                default (triggered by any negative number) is three times that
                or 100, whichever is greater.
            recordAfter: How many times the sampler must be called before the
                adapted covariance begins to take in samples.  This is to
                prevent pre-burned-in samples from being used to adapt with,
                which can dramatically reduce the effectiveness of sampling.
            refreshPeriod: How many times the sampler is called between the
                cholesky of the covariance being updated (the expensive part).
            scaling:  How much to scale the estimated covariance to get the 
                proposal covariance.  Default, signified by -1, is to use
                5.76/nDim, suggested in Haario et al.
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
            adaptAfter = max(3*(dStop-dStart),100)
        if recordAfter < 0:
            recordAfter = <int>(adaptAfter/2.)
        assert recordAfter < adaptAfter, "Must begin recording before adaptation can begin."
        assert dStop-dStart < adaptAfter, "Must have enough samples to compute the covariance before adaptation can begin."
        if scaling <= 0:
            scaling = 5.74/self.nDim
        if covariance is None:
            covariance = np.diag(np.asarray(self._scale)[dStart:dStop])**2
        assert covariance.shape[0] == covariance.shape[1] == dStop-dStart, "Misshapen covariance."
        cdef SamplerData samp
        covChol = cholesky(covariance)
        samp.samplerType = 3
        samp.dStart = dStart
        samp.dStop = dStop
        samp.idata.push_back(0)
        samp.idata.push_back(adaptAfter)
        samp.idata.push_back(recordAfter)
        samp.idata.push_back(refreshPeriod)
        samp.ddata = np.concatenate([[eps,scaling],np.zeros(covariance.shape[0]),
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

    cpdef object enableSurrogate(self, xInit, yInit, kernel='matern32', tol=1e-2):
        '''Turns on surrogate sampling and initializes the surrogate GP.

        In order to get reasonable parameters for the Gaussian process, you
        must provide at least some initial points to optimize the GP on.
        More points can help reduce the number of likelihood evaluations, but
        will slow down GP computation, so don't overdo it.

        Args:
            xInit: The sample positions to initialize the GP with.
            yInit: The values of the likelihood to initialize the GP with.
            kernel: The kernel to use in the GP surrogate.  Must be
                differentiable if you want to use the gradient.
            tol: How much uncertainty in the log likelihood to permit without
                calling the likelihood function.
        '''
        if self.useSurrogate:
            raise ValueError("Surrogate sampling is already enabled.")
        scaledPosition = xInit/np.asarray(self._scale)[None,:]
        self.surrogateOffset = np.median(yInit)
        self.surrogate = GaussianProcess(scaledPosition,yInit-self.surrogateOffset,kernel)
        self.surrogate.optimizeParams()
        self.surrogateTol = tol
        self.surrogateSamples = len(yInit)
        self.surrogateLastOptimize = self.surrogateSamples
        self.useSurrogate = True
        return

    cpdef object disableSurrogate(self):
        '''Disables the use of a surrogate model if previously enabled.
        '''
        if not self.useSurrogate:
            raise ValueError("Surrogate sampling is already disabled.")
        self.surrogate = None
        self.useSurrogate = False
        return

    cpdef object getSampler(self, unsigned int i=0):
        assert i < self.samplers.size()
        return self.samplers[i]

    cpdef object getProposalCov(self, unsigned int i=0):
        assert i < self.samplers.size()
        if self.samplers[i].samplerType == 0:
            return np.diag(self._scale)
        elif self.samplers[i].samplerType == 1:
            return np.diag(1./np.asarray(self._scale))
        elif self.samplers[i].samplerType == 2:
            n = self.samplers[i].dStop - self.samplers[i].dStart
            chol = np.asarray(self.samplers[i].ddata).reshape(n,n)
            return np.matmul(chol,chol.T)
        elif self.samplers[i].samplerType == 3:
            n = self.samplers[i].dStop - self.samplers[i].dStart
            return np.asarray(self.samplers[i].ddata[2+n:2+n+n**2]).reshape(n,n)


    cpdef object printSamplers(self):
        '''Prints the list of any/all sampling systems set up so far.

        This refers to samplers added using e.g. addMetropolis() functions. See
        the overall class documentation for more information on these.  The
        type of function and any parameters given (unique to each sampler
        type) are printed.
        '''
        cdef size_t s
        if self.samplers.size() == 0:
            print("No samplers defined.")
        for s in range(self.samplers.size()):
            if self.samplers[s].samplerType == 0:
                print(s, "Diagonal Metropolis ("+str(self.samplers[s].dStart)+":"+str(self.samplers[s].dStop)+")")
                print('\tdiag(C) =', np.asarray(self._scale))
            elif self.samplers[s].samplerType == 1:
                print(s, "HMC ("+str(self.samplers[s].dStart)+":"+str(self.samplers[s].dStop)+"), ",\
                    self.samplers[s].idata[0], "steps with size", self.samplers[s].ddata[0])
            elif self.samplers[s].samplerType == 2:
                print(s, "Metropolis ("+str(self.samplers[s].dStart)+":"+str(self.samplers[s].dStop)+"), Cov =")
                n = self.samplers[s].dStop - self.samplers[s].dStart
                a = np.asarray(self.samplers[s].ddata).reshape(n,n)
                print(np.matmul(a,a.T))
            elif self.samplers[s].samplerType == 3:
                print(s, "Adaptive Metropolis ("+str(self.samplers[s].dStart)+":"+str(self.samplers[s].dStop)+"),",\
                    "Start adapting after", self.samplers[s].idata[1], \
                    "steps, updating every", self.samplers[s].idata[2], "steps.")
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
            self.samplesLogProbView[i] = self.lastLogProb
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

    cpdef object run(self, Size nSamples, x, Size burnIn=0, Size thinning=0, Size recordStart=0, Size recordStop=-1, bint collectStats=False, Size threads=1, bint showProgress=True):
        '''Begin sampling the parameters from the given logProbability dist.

        Args:
            nSamples: The desired number of samples to record per thread.
            x: The initial position(s) to start the sampler at.
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
        x0 = np.atleast_1d(x).astype(np.double)
        assert nSamples > 0, "The number of samples must be greater than 0."
        assert threads > 0, "Threads must be > 0."
        assert (x0.shape == (self.nDim,) or x0.shape == (threads,self.nDim)), "The initial guess must have shape [nDim], or [threads, nDim]."
        self.initialPosition = x0.copy()
        self.showProgress = showProgress
        self.recordStart = recordStart
        if recordStop < 0:
            recordStop = self.nDim
        self.recordStop = recordStop
        self.accepted = np.zeros(self.nDim,dtype=np.intc)
        self.acceptedView = self.accepted
        self.nSamples = nSamples
        self.burnIn = burnIn
        self.thinning = thinning
        self.sampleStats.clear()
        self.lastLogProb = nan
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
                self.samples, self.samplesLogProb, self.accepted = zip(*p.map_async(self,list(x0)).get(1000000000))
            except KeyboardInterrupt:
                print("ERROR: Sampling terminated by keyboard interrupt.")
                self.readyToRun=False
            except Exception as e:
                print("ERROR: Sam parent thread encountered an exception:", e.message)
                self.readyToRun=False
            p.terminate()
            p.close()
            if self.readyToRun:
                self.samples = np.array(self.samples)
                self.samplesLogProb = np.array(self.samplesLogProb)
                self.accepted = np.array(self.accepted)
                self.results = np.reshape(self.samples,(threads*self.nSamples,self.nDim))
                self.resultsLogProb = np.reshape(self.samplesLogProb,(threads*self.nSamples))
                self.readyToRun = False
                return self.samples
            return None
        else:
            self(x0)
            self.results = self.samples
            self.resultsLogProb = self.samplesLogProb
            return self.samples

    def __call__(self, double[:] x0):
        '''Internal function used to run the sampler.  Not for user use!

        This function is an internal way to sample, which assumes certain
        other internal parameters have already been set up.  Users should call
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
        self.samplesLogProb = np.empty((self.nSamples),dtype=np.double)
        self.sampleView = self.samples
        self.samplesLogProbView = self.samplesLogProb
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
            print("")
        return self.samples, self.samplesLogProb, self.accepted

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

    cpdef object getDIC(self,prior):
        '''Approximates the DIC from the sampled values.

        This function needs to know the prior because only the full posterior
        probability was given to the sampler at runtime.  The value of the
        prior is subtracted off, and the sample with the maximum resulting
        likelihood is identified.  The BIC is computed assuming that this is
        the global maximum likelihood; for a well sampled posterior, this will
        be a good approximation.

        Args:
            prior: a function which takes an array of parameters (same length
                as the logProbability function) and returns the prior for them.

        Returns:
            The estimated DIC for the model as a floating point number.
        '''
        assert self.trials > 0, "The number of trials must be greater than zero to compute a DIC."
        l = np.array([logP-prior(theta) for logP, theta in zip(self.resultsLogProb,self.results)])
        return getDIC(l)

    cpdef object getAIC(self,prior):
        '''Approximates the AIC from the sampled values.

        This function needs to know the prior because only the full posterior
        probability was given to the sampler at runtime.  The value of the
        prior is subtracted off, and the sample with the maximum resulting
        likelihood is identified.  The BIC is computed assuming that this is
        the global maximum likelihood; for a well sampled posterior, this will
        be a good approximation.

        This function shouldn't be used in a heirarchical setting, as the AIC
        is not defined for that case (the number of parameters isn't clearly
        defined).  Consider the DIC instead.

        Args:
            prior: a function which takes an array of parameters (same length
                as the logProbability function) and returns the prior for them.

        Returns:
            The estimated AIC for the model as a floating point number.
        '''
        assert self.trials > 0, "The number of trials must be greater than zero to compute a AIC."
        lMax = max([logP-prior(theta) for logP, theta in zip(self.resultsLogProb,self.results)])
        return getAIC(lMax,self.nDim)

    cpdef object getBIC(self,prior,nPoints):
        '''Approximates the BIC from the sampled values.

        This function needs to know the prior because only the full posterior
        probability was given to the sampler at runtime.  The value of the
        prior is subtracted off, and the sample with the maximum resulting
        likelihood is identified.  The BIC is computed assuming that this is
        the global maximum likelihood; for a well sampled posterior, this will
        be a good approximation.

        This function shouldn't be used in a heirarchical setting, as the BIC
        is not defined for that case (the number of parameters isn't clearly
        defined).  Consider the DIC instead.

        Args:
            prior: a function which takes an array of parameters (same length
                as the logProbability function) and returns the prior for them.
            nPoints: The number of data points used to evaluate the likelihood.

        Returns:
            The estimated BIC for the model as a floating point number.
        '''
        assert self.trials > 0, "The number of trials must be greater than zero to compute a BIC."
        lMax = max([logP-prior(theta) for logP, theta in zip(self.resultsLogProb,self.results)])
        return getBIC(lMax,self.nDim,nPoints)

    cpdef object summary(self, paramIndices=None, returnString=False):
        '''Prints/returns some summary statistics of the previous sampling run.

        Statistics are the parameter index, the acceptance rate, mean, and
        standard deviation, as well as the 16th, 50th, and 84th percentiles.

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
        grs = gelmanRubin(self.samples,False)
        grLabel = "GR"
        if np.ndim(self.samples) == 2:
            grLabel = "GR*"
        percents = np.percentile(self.results,(16,50,84),axis=0)
        output = ("{:<4}"+" {:>6}"+" {:>6}"+" |"+2*" {:>10}"+" |"+3*" {:>10}").format("Dim.","Accept",grLabel,"Mean","Std.","16%","50%","84%")
        for i in paramIndices:
            output += '\n' + ("{:<4}"+" {:>6.1%}"+" {:>6.3f}" + " |"+2*" {:>10.4g}"+" |"+3*" {:>10.4g}").format(i,acceptance[i],grs[i],means[i],stds[i],percents[0,i],percents[1,i],percents[2,i])
        output += '\n'
        if returnString:
            return output
        print(output)
        return

    @property
    def userParams(self):
        return self._userParameters

    @userParams.setter
    def userParams(self, params):
        self._userParameters = np.asarray(params, np.double, "C").flatten().copy()
        self.userParametersView = self._userParameters

    @property
    def scale(self):
        return np.asarray(self._scale).copy()

    @scale.setter
    def scale(self, newScale):
        newScale = np.asarray(newScale, np.double, "C").flatten()
        assert len(newScale) == self.nDim, "New scale must have the same length as the sampler dimensions."
        assert all(newScale > 0.), "Scale must be positive!"
        assert all(np.isfinite(newScale)), "Scale must be finite!"
        for d in range(self.nDim):
            self._scale[d] = newScale[d]

    @property
    def lowerBounds(self):
        if self.lowerBoundaries is None:
            return None
        else:
            return np.asarray(self.lowerBoundaries).copy()

    @lowerBounds.setter
    def lowerBounds(self, newBounds):
        assert len(newBounds) == self.nDim, "New scale must have the same length as the sampler dimensions."
        assert not any(np.isnan(newBounds)), "Scale must be not be NaN (infinite is fine)."
        if self.upperBoundaries is not None:
            assert all(np.asarray(self.upperBoundaries) > newBounds), "New lower bounds must be less than upper bounds."
        for d in range(self.nDim):
            self.lowerBoundaries[d] = newBounds[d]

    @property
    def upperBounds(self):
        if self.upperBoundaries is None:
            return None
        else:
            return np.asarray(self.upperBoundaries).copy()

    @upperBounds.setter
    def upperBounds(self, newBounds):
        assert len(newBounds) == self.nDim, "New scale must have the same length as the sampler dimensions."
        assert not any(np.isnan(newBounds)), "Scale must be not be NaN (infinite is fine)."
        if self.lowerBoundaries is not None:
            assert all(np.asarray(self.upperBoundaries) > newBounds), "New upper bounds must be greater than lower bounds."
        for d in range(self.nDim):
            self.upperBoundaries[d] = newBounds[d]

    def getACF(self, length=50):
        assert self.trials > 0, "The number of trials must be greater than zero to compute the autocorrelation function."
        if self.samples.ndim == 2:
            return acf(self.samples)
        if self.samples.ndim == 3:
            return np.array([acf(self.samples[i,:,i]) for i in range(self.samples.shape[0])])

    cpdef object testGradient(self, x, double eps=1e-5):
        '''Compares gradients from logProbability and finite difference method

        This function computes a gradient estimate using the finite difference
        method and compares it to the gradient computed by the logProbability
        function (specified at initialization).

        Args:
            x: The central location to compute the gradient at.
            eps: scales the finite difference method.  Secondary samples are
                taken in the direction of each parameter a distance scale*eps
                away.

        Returns:
            The relative difference between the logProbability gradient L and
            the finite difference gradient F: (L-F)/L+F)
        '''
        cdef double[:] x0 = np.atleast_1d(x).astype(np.double)
        assert x0.size == self.nDim, "The starting position given has wrong number of dimensions."
        cdef Size d
        for d in range(self.nDim):
            self.x[d] = x0[d]
        cdef double central = self.logProbability(x0,self.gradient,True)
        cdef double estimate
        cdef object output = np.empty(self.nDim,dtype=np.double)
        cdef double[:] outputView = output
        for d in range(self.nDim):
            x0[d] += self._scale[d]*eps
            estimate = (self.logProbability(x0,self.momentum,False) - central)/(self._scale[d]*eps)
            try:
                outputView[d] = (estimate-self.gradient[d])/(estimate+self.gradient[d])
            except:
                outputView[d] = nan
            x0[d] -= self._scale[d]*eps
        return output

    cpdef object gradientDescent(self, x, double step=.1, double eps=1e-10):
        cdef double[:] x0 = np.atleast_1d(x).astype(np.double)
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
                xNew = self.x[d] + step*self._scale[d]*self._scale[d]*self.gradient[d]
                if xNew < self.lowerBoundaries[d]:
                    xNew = self.lowerBoundaries[d]
                if xNew > self.upperBoundaries[d]:
                    xNew = self.upperBoundaries[d]
                if abs(xNew - self.x[d]) > abs(eps * self._scale[d]):
                    done = False
                self.x[d] = xNew
        output = np.zeros(self.nDim)
        for d in range(self.nDim):
            output[d] = self.x[d]
        return output

    cpdef object simulatedAnnealing(self, x, Size nSteps=200, Size nQuench=200, double T0=5, double width=1.0):
        cdef double[:] x0 = np.atleast_1d(x).astype(np.double)
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
                self.xPropose[d] = normalRand(self.x[d],width*self._scale[d])
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
                self.xPropose[d] = normalRand(self.x[d],width*self._scale[d]/5.)
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
        '''
        f = (10*i)//N
        sys.stdout.write('\r'+header+': <'+f*"="+(10-f)*" "+'> ('+str(i)+" / " + str(N) + ")          ")
        sys.stdout.flush()
        return 0

    cdef int onlineCovar(self, double[:] mu, double[:,:] covar, double[:] x, int t, double scaling, double eps=1e-9) except -1:
        # TODO: Documentation
        cdef Size i, j
        cdef Size n = mu.shape[0]
        # TODO: make more efficient?
        cdef double[:] muOld = mu.copy()
        for i in range(n):
            mu[i] = (t*mu[i]+x[i])/(t+1)
        if t > 0:
            for i in range(n):
                for j in range(n):
                    covar[i,j] = (t-1)*covar[i,j] + scaling*(t*muOld[i]*muOld[j] - (t+1)*mu[i]*mu[j] + x[i]*x[j])
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
        self._scale = self._workingMemory_[4*self.nDim:5*self.nDim]
        self.upperBoundaries = self._workingMemory_[5*self.nDim:6*self.nDim]
        self.lowerBoundaries = self._workingMemory_[6*self.nDim:7*self.nDim]
        self.acceptedView = self.accepted
        self.userParametersView = self._userParameters
        return 0

    def __init__(self, logProbability, scale, lowerBounds=None, upperBounds=None, extraMembers=[]):
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
            extraMembers: If a subclass defines aditional members, listing their
                names here (list of strings) will cause them to be handled
                correctly when multiple threads are used, and available to save
                in Sam.save.

        Returns:
            An instantiated object of the class.
        '''
        scale = np.atleast_1d(scale).astype(np.double)
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
        self.extraMembers = extraMembers
        self._workingMemory_ = np.nan * np.ones(7*self.nDim,dtype=np.double)
        self.nSamples = 0
        self.accepted = np.zeros(self.nDim,dtype=np.intc)
        self.trials = 0
        self._userParameters = np.array([])
        self._setMemoryViews_()
        self.pyLogProbability = logProbability
        if self.pyLogProbability is not None:
            if (sys.version_info > (3, 0)):
                self.pyLogProbArgNum = len(inspect.signature(self.pyLogProbability).parameters)
            else:
                self.pyLogProbArgNum = len(inspect.getargspec(self.pyLogProbability).args)
            assert self.pyLogProbArgNum in [1,2,3,4], "The logProbability function must take one to four arguments."
        else:
            self.pyLogProbArgNum = -1
        for d in range(self.nDim):
            self._scale[d] = scale[d]
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


    def laplaceApprox(self, initialGuess, newSampler=None, computeSamples=0, warnPrecision=False, optArgs={}):
        '''Applies the Laplace approximation to the posterior for better initialization.

        The Laplace approximation efficiently finds a multivariate normal by setting the mean
        to the posterior maximum (found numerically) and the covariance as the inverse hessian of
        the posterior at that point.  If requested, this function can use this information to set
        a good initial guess for the correlated Metropolis sampler or the adaptive sampler.

        Args:
            initialGuess: The starting point for the optimization.  Must be in bounds and finite.
            newSampler: What to configure the samplers to "metropolis" or "adaptive".  If None
                (the default), the samplers will not be changed.
            computeSamples: If nonzero, the function will also fill the results array from the
                approximate multivariate normal, as if sampling had been run.

        Returns:
            The mean (1-d array) and covariance (2-d array) estimated for the posterior.
        '''
        initialGuess = np.atleast_1d(initialGuess).astype(np.double)

        if np.any(initialGuess <= self.lowerBoundaries) or np.any(initialGuess >= self.upperBoundaries):
            raise ValueError("Initial guess was not inside the parameter bounds!")
        if not np.isfinite(self.logProbability(initialGuess, initialGuess, False)):
            raise ValueError("Initial guess was NaN or infinite!")
        assert newSampler is None or newSampler in ["metropolis", "adaptive"], "Invalid newSampler argument."

        def target(x):
            # Impose soft boundary conditions for the optimizer, with a gradient pointing back in bounds
            penalty = 0.
            for i in range(self.nDim):
                if x[i] <= self.lowerBoundaries[i]:
                    penalty -= 5. * abs(1. + (self.lowerBoundaries[i]-x[i]) / self._scale[i])
                elif x[i] >= self.upperBoundaries[i]:
                    penalty -= 5. * abs(1. + (x[i]-self.upperBoundaries[i]) / self._scale[i])
            x = np.clip(x, self.lowerBoundaries, self.upperBoundaries)
            result = -(self.logProbability(x, x, False) + penalty)
            return result if np.isfinite(result) else 1e12

        # Run once to ensure we're starting near the max, then again to get the inverse hessian
        result = minimize(target, initialGuess, method="SLSQP", options={'ftol':0.1})
        result = minimize(target, result.x, **optArgs)

        if not result.success:
            if result.status == 2:
                if warnPrecision:
                    print("Warning:", result.message)
            else:
                print("ERROR:", result.status, result.message)
                raise ValueError("Code " + str(result.status) + ". "+ result.message)

        self.initialPosition = result.x

        if newSampler is not None:
            if newSampler == "metropolis":
                self.clearSamplers()
                self.addMetropolis(result.hess_inv * 5.74 / self.nDim)
            elif newSampler == "adaptive":
                self.clearSamplers()
                self.addAdaptiveMetropolis(
                    result.hess_inv, adaptAfter=500, recordAfter=100, refreshPeriod=100)

        if computeSamples > 0:
            self.nSamples = computeSamples
            self.trials = computeSamples
            self.samples = np.atleast_2d(multivariate_normal.rvs(result.x, result.hess_inv, size=computeSamples).T).T
            self.results = self.samples
            self.samplesLogProb = np.nan*np.empty((self.nSamples),dtype=np.double)
            self.resultsLogProb = self.samplesLogProb
            for i in range(self.nDim):
                self.accepted[i] = computeSamples

        return result.x, result.hess_inv


    def save(self, filename, extraMembers=True):
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
            extraMembers: saves extra subclass members listed during the __init__;

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
        if extraMembers:
            extraMembers = {}
            for i in self.extraMembers:
                attr = getattr(self, i)
                if np.iterable(attr):
                    attr = np.array(attr)
                extraMembers[i] = attr
        else:
            extraMembers = None
        try:
            logProbSource = inspect.getsource(self.pyLogProbability)
        except:
            logProbSource = "Source code not available."
        np.savez_compressed(
            filename, nDim=self.nDim, nSamples=self.nSamples, burnIn = self.burnIn,
            thinning=self.thinning, scale=np.asarray(self._scale), upperBounds=self.upperBoundaries,
            lowerBounds=self.lowerBoundaries, initialPosition=self.initialPosition,
            samples = self.samples, acceptance = accept, logProbSource = logProbSource,
            samplesLogProb = self.samplesLogProb, userParameters=self.userParameters, stats=stats, **extraMembers)

    def __getstate__(self):
        '''Prepares internal memory for pickling.

        This is important for compatibility with the multiprocessing library
        used for parallelism.

        Returns:
            A pickleable tuple of the internal variables.
        '''
        extra = {}
        for i in self.extraMembers:
            attr = getattr(self, i)
            if np.iterable(attr):
                attr = np.array(attr)
            extra[i] = attr
        info = (self.nDim, self.nSamples, self.burnIn, self.thinning, self.recordStart,
                self.recordStop, self.collectStats, self.readyToRun, self.samplers, self.lastLogProb,
                self._workingMemory_, self.accepted, self.pyLogProbability, self.pyLogProbArgNum,
                self.hasBoundaries, self.showProgress, self._userParameters, extra)
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
         self.recordStop, self.collectStats, self.readyToRun, self.samplers, self.lastLogProb,
         self._workingMemory_, self.accepted, self.pyLogProbability, self.pyLogProbArgNum,
         self.hasBoundaries, self.showProgress, self._userParameters, extraMembers) = info
        for k, v in extraMembers.items():
            setattr(self, k, v)
        if (sys.version_info > (3, 0)):
            defaultEngine.setSeed(<unsigned long int>int(os.urandom(4).hex(),16))
            np.random.seed(int(os.urandom(4).hex(),16))
        else:
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
