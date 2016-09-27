include "distributions.pyx"
import numpy as np
import time
cimport numpy as np

cpdef void test():
    cdef Size i
    for i in range(10):
        print Normal.rand(1,.2), Gamma.rand(1,1), Beta.rand(2,2)
    return

Normal = _normal(<unsigned long int>(1*1000*time.time()))
Gamma = _gamma(<unsigned long int>(2*1000*time.time()))
InvGamma = _invGamma(<unsigned long int>(3*1000*time.time()))
InvChiSq = _invChiSq(<unsigned long int>(4*1000*time.time()))
Beta = _beta(<unsigned long int>(5*1000*time.time()))


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
