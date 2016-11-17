# distutils: language = c++
include "distributions.pyx"
include "griddy.pyx"
import numpy as np
cimport numpy as np

# Special function wrappers
cpdef double incBeta(double x, double a, double b):
    return _incBeta(a,b,x)

cdef class Sam:
    cpdef double logProbability(self, double[:] position):
        if self.pyLogProbability is None:
            raise NotImplementedError("You haven't defined the log probability,"+
                                      "but the sampler called it.")
        return self.pyLogProbability(np.asarray(position))

    cpdef void gradLogProbability(self, double[:] position, double[:] output):
        if self.pyGradLogProbability is None:
            raise NotImplementedError("You haven't defined the log probability "+
                                      "gradient, but the sampler called it.")
        np.asarray(output)[:] = self.pyGradLogProbability(np.asarray(position))

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
        # Initialize velocities
        for d in range(dStart,dStop):
            self.xPropose[d] = self.x[d]
            self.momentum[d] = normalRand(0,1./sqrt(self.scale[d]))

        # Compute the kinetic energy part of the initial Hamiltonian
        cdef double kinetic = 0
        for d in range(dStart,dStop):
            kinetic += self.momentum[d]*self.momentum[d]*self.scale[d] / 2.0

        # Simulate the trajectory
        self.gradLogProbability(self.xPropose,self.gradient)
        for i in range(nSteps):
            for d in range(dStart,dStop):
                self.momentum[d] += stepSize * self.gradient[d] / 2.0
            self.bouncingMove(stepSize, dStart, dStop)
            self.gradLogProbability(self.xPropose, self.gradient)
            for d in range(dStart,dStop):
                self.momentum[d] += stepSize * self.gradient[d] / 2.0

        # Compute the kinetic energy part of the proposal Hamiltonian
        cdef double kineticPropose = 0
        for d in range(dStart,dStop):
            kineticPropose += self.momentum[d]*self.momentum[d]*self.scale[d]/2.0

        # Decide whether to accept the new point
        cdef double old = self.logProbability(self.x)
        cdef double new = self.logProbability(self.xPropose)
        if isnan(new) or isnan(old):
            raise ValueError("Got NaN for the log probability!")
        if (-exponentialRand(1.) < new - kineticPropose - old + kinetic):
            self.acceptanceRate += 1.
            for d in range(dStart,dStop):
                self.x[d] = self.xPropose[d]
        return

    cdef void bouncingMove(self, double stepSize, Size dStart, Size dStop):
        cdef Size d
        for d in range(dStart,dStop):
            self.xPropose[d] += self.momentum[d] * stepSize * self.scale[d]
            # Enforce boundary conditions
            while True:
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
        for d in range(dStart,dStop):
            self.xPropose[d] = self.x[d] + normalRand(0,self.scale[d])
            if(self.xPropose[d] > self.upperBoundaries[d] or
               self.xPropose[d] < self.lowerBoundaries[d]):
                return
        for d in range(0,dStart):
            self.xPropose[d] = self.x[d]
        for d in range(dStop,self.nDim):
            self.xPropose[d] = self.x[d]
        if (-exponentialRand(1.) < self.logProbability(self.xPropose) -
            self.logProbability(self.x)):
            self.acceptanceRate += 1.
            for d in range(dStart,dStop):
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
        cdef SamplerData samp
        samp.samplerType = 0
        samp.dStart = dStart
        samp.dStop = dStop
        self.samplers.push_back(samp)

    cpdef void addHMC(self, Size nSteps, double stepSize, Size dStart, Size dStop):
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
        for d in range(self.nDim):
            self.sampleView[i,d] = self.x[d]
        return

    cpdef object run(self, Size nSamples, double[:] x0, Size burnIn=0, Size thinning=0):
        cdef Size i, j, d
        cdef double vMag, vMagPropose
        self.acceptanceRate = 0
        self.samples = np.empty((nSamples,self.nDim),dtype=np.double)
        self.sampleView = self.samples
        if not self.samplers.size():
            print "No samplers defined -- defaulting to metropolis."
            self.addMetropolis(0,self.nDim)
        for d in range(self.nDim):
            self.x[d] = x0[d]
        for i in range(nSamples+burnIn):
            for j in range(thinning+1):
                self.sample()
            if i >= burnIn: 
                self.record(i-burnIn)
                # TODO: track statistics here
        self.acceptanceRate /= (nSamples + burnIn) * (thinning+1)
        return self.samples


    cpdef void testGradient(self, double[:] x0, double eps=1e-5):
        cdef double central = self.logProbability(x0)
        cdef double estimate
        self.gradLogProbability(x0,self.gradient)
        for d in range(self.nDim):
            x0[d] += self.scale[d]*eps
            estimate = (self.logProbability(x0) - central)/(self.scale[d]*eps)
            print d, (estimate-self.gradient[d])/(estimate+self.gradient[d])
            x0[d] -= self.scale[d]*eps
        return

    cpdef object gradientDescent(self, double[:] x0, double step=.1, double eps=1e-10):
        cdef Size d, i
        cdef bint done = 0
        cdef double xNew
        for d in range(self.nDim):
            self.x[d] = x0[d]
        while not done:
            self.gradLogProbability(self.x,self.gradient)
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
        cdef Size d, i
        cdef double energy, energyPropose, temperature
        for d in range(self.nDim):
            self.x[d] = x0[d]
        energy = self.logProbability(self.x)
        for i in range(nSteps):
            temperature = T0*(1. - (<double>i)/(nSteps))
            for d in range(self.nDim):
                self.xPropose[d] = normalRand(self.x[d],width*self.scale[d])
            energyPropose = self.logProbability(self.xPropose)
            if (energyPropose - energy)/temperature > -exponentialRand(1.):
                for d in range(self.nDim):
                    self.x[d] = self.xPropose[d]
                    energy = energyPropose
        for i in range(nQuench):
            for d in range(self.nDim):
                self.xPropose[d] = normalRand(self.x[d],width*self.scale[d]/5.)
            energyPropose = self.logProbability(self.xPropose)
            if (energyPropose > energy):
                for d in range(self.nDim):
                    self.x[d] = self.xPropose[d]
                    energy = energyPropose
        output = np.zeros(self.nDim)
        for d in range(self.nDim):
            output[d] = self.x[d]
        return output

    def __init__(self, object logProbability, Size nDim, double[:] scale, double[:] upperBoundaries=None, double[:] lowerBoundaries=None, object gradLogProbability = None):
        cdef Size d
        self.nDim = nDim
        self.x = np.zeros(self.nDim,dtype=np.double)
        self.momentum = np.empty(self.nDim,dtype=np.double)
        self.xPropose = np.zeros(self.nDim,dtype=np.double)
        self.gradient = np.empty(self.nDim,dtype=np.double)
        self.scale = np.empty(self.nDim,dtype=np.double)
        self.upperBoundaries = np.empty(self.nDim,dtype=np.double)
        self.lowerBoundaries = np.empty(self.nDim,dtype=np.double)
        self.pyLogProbability = logProbability
        self.pyGradLogProbability = gradLogProbability
        for d in range(self.nDim):
            self.scale[d] = scale[d]
        if upperBoundaries is not None:
            for d in range(self.nDim):
                self.upperBoundaries[d] = upperBoundaries[d]
        else:
            for d in range(self.nDim):
                self.upperBoundaries[d] = infinity
        if lowerBoundaries is not None:
            for d in range(self.nDim):
                self.lowerBoundaries[d] = lowerBoundaries[d]
        else:
            for d in range(self.nDim):
                self.lowerBoundaries[d] = -infinity
        return
