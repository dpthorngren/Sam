import numpy as np
from scipy.interpolate.interpnd import _ndim_coords_from_arrays

# TODO: Fix for 2 points grids
cdef class Griddy:
    def __init__(self,axes,values):
        cdef Size d, i, totalSize, maxPoints
        self.nDim = len(axes)
        cdef bint[:] uniform = np.empty(self.nDim,dtype=np.intc)
        self.nPoints = np.empty(self.nDim,dtype=int)
        strides = np.ones(self.nDim,dtype=int)
        totalSize = 1
        maxPoints = 4
        for d in range(self.nDim):
            # Sanity check
            assert np.isfinite(axes[d]).all()
            assert (np.diff(axes[d])>0.0).all()
            self.nPoints[d] = len(axes[d])
            totalSize *= self.nPoints[d]
            # Check for uniformity
            if np.allclose(axes[d],
                           np.linspace(axes[d][0], axes[d][-1], axes[d].size),
                           rtol=1e-6):
                uniform[d] = True
            else:
                uniform[d] = False
                if maxPoints < self.nPoints[d]:
                    maxPoints = self.nPoints[d]
        for d in range(self.nDim):
            if d > 0:
                strides[:d] *= self.nPoints[d]

        self.strides = strides
        self.axes = np.empty((self.nDim,maxPoints),dtype=np.double)
        for d in range(self.nDim):
            if uniform[d]:
                # If a variable is uniform, record min, nan, stepSize
                self.axes[d,0] = axes[d].min()
                self.axes[d,1] = nan
                self.axes[d,2] = (axes[d].max() - axes[d].min())/(self.nPoints[d]-1)
                self.axes[d,3] = axes[d].max()
                continue
            for i in range(maxPoints):
                if i < self.nPoints[d]:
                    self.axes[d,i] = axes[d][i]
                else:
                    self.axes[d,i] = np.inf
        assert values.size == totalSize
        self.values = values.flatten().copy()

        # Allocate working memory
        self.weights = np.empty((self.nDim),dtype=np.double)
        self.widths = np.empty((self.nDim),dtype=np.double)
        self.indices = np.empty((self.nDim),dtype=int)
        self.tempIndices = np.empty((self.nDim),dtype=int)
        return

    cpdef object getValues(self):
        return np.asarray(self.values).copy()

    cpdef object getNPoints(self):
        return np.asarray(self.nPoints).copy()

    cpdef object getIndices(self):
        return np.asarray(self.indices).copy()

    cpdef object getWeights(self):
        return np.asarray(self.weights).copy()

    cpdef object getStrides(self):
        return np.asarray(self.strides).copy()

    cpdef Size ind(self,Size[:] p):
        cdef Size d, index = 0
        for d in range(self.nDim):
            index += p[d]*self.strides[d]
        return index

    cpdef bint locatePoints(self, double[:] point):
        # Locates the indices corresponding to the point
        # Writes to self.indices, returns whether out of bounds
        cdef Size d, i, low, high
        cdef bint outOfBounds = False
        # Start with a bounds check
        for d in range(self.nDim):
            if isnan(self.axes[d,1]):
                if point[d] < self.axes[d,0]:
                    self.indices[d] = 0
                    outOfBounds = True
                    continue
                if point[d] >= self.axes[d,3]:
                    self.indices[d] = self.nPoints[d]-2
                    outOfBounds = True
                    continue
            else:
                if point[d] < self.axes[d,0]:
                    self.indices[d] = 0
                    outOfBounds = True
                    continue
                if point[d] >= self.axes[d,self.nPoints[d]-1]:
                    self.indices[d] = self.nPoints[d]-2
                    outOfBounds = True
                    continue
            # Check for uniformity (flagged with axes[d,1] == nan)
            if self.axes[d,1] != self.axes[d,1]:
                i = <Size>((point[d]-self.axes[d,0])/self.axes[d,2])
                self.indices[d] = i
                self.widths[d] = self.axes[d,2]
                #TODO: fix this 1e-10 hack
                self.weights[d] = (point[d] - (self.axes[d,0] + i*self.axes[d,2])) / self.widths[d] + 1e-10
            else:
                low = 0
                high = self.nPoints[d]-1
                i = (low+high)/2
                # Binary search for the correct indices
                while not (self.axes[d,i] <= point[d] < self.axes[d,i+1]):
                    i = (low+high)/2
                    if point[d] > self.axes[d,i]:
                        low = i
                    else:
                        high = i
                self.indices[d] = i
                self.widths[d] = (self.axes[d,i+1]-self.axes[d,i])
                self.weights[d] = (point[d] - self.axes[d,i]) / self.widths[d] + 1e-10
        return outOfBounds

    def __call__(self,x,gradient=False,debug=False):
        # Using the RegularGridInterpolator system for sanity
        cdef Size d, i, nInputs
        cdef double[:,:] gradsView
        cdef double[:,:] inputs = np.atleast_2d(_ndim_coords_from_arrays(x, ndim=self.nDim))
        if inputs.shape[1] != self.nDim:
            raise ValueError("Wrong number of dimensions in argument.")
        nInputs = inputs.shape[0]
        outputs = np.empty(nInputs,dtype=np.double)
        cdef double[:] outputsView = outputs
        if gradient:
            grads = np.empty((nInputs,self.nDim),dtype=np.double)
            gradsView = grads
            for i in range(nInputs):
                outputsView[i] = self.interp(inputs[i],gradsView[i],locate=True,debug=1)
            if nInputs == 1:
                return outputs[0], grads[0]
            return outputs, grads
        else:
            for i in range(nInputs):
                outputsView[i] = self.interp(inputs[i])
            if nInputs==1:
                return outputs[0]
            return outputs

    cpdef double interp(self,double[:] points, double [:] gradient=None, bint locate=True, bint debug=False):
        # Locate indices and compute weights, or return nan
        cdef Size b, d, offset
        if debug:
            print "Input:",
            for d in range(self.nDim):
                print points[d],
            print ""
        if locate:
            if(self.locatePoints(points)):
                if debug:
                    print "\tOut of Bounds: ",
                    for d in range(self.nDim):
                        print self.indices[d],
                    print ""
                return nan
        if debug:
            print "\tIndices:",
            for d in range(self.nDim):
                print self.indices[d],
            print ""
            print "\tWeights:",
            for d in range(self.nDim):
                print self.weights[d],
            print ""
        cdef double result = 0.0
        cdef double netWeight, gradWeight, adjustment
        if gradient is not None:
            for d in range(self.nDim):
                gradient[d] = 0

        # Sum over bounding points
        for b in range(1<<self.nDim):
            netWeight = 1.0
            # Compute total weight for each bounding point
            for d in range(self.nDim):
                offset = b>>d&1
                self.tempIndices[d] = self.indices[d]+offset
                if offset:
                    netWeight *= self.weights[d]
                else:
                    netWeight *= 1.-self.weights[d]
            adjustment = netWeight*self.values[self.ind(self.tempIndices)]
            if debug:
                print "\tPoint {0} (".format(b),
                for d in range(self.nDim):
                    print self.indices[d] + (b>>d&1),
                print "): {0}, {1}".format(netWeight,self.values[self.ind(self.tempIndices)])
            result += adjustment
            if gradient is not None:
                for d in range(self.nDim):
                    offset = b>>d&1
                    if offset:
                        gradient[d] += adjustment / (self.weights[d]*self.widths[d])
                    else:
                        gradient[d] -= adjustment / ((1-self.weights[d])*self.widths[d])
        if debug:
            print "\tOutput: {0}".format(result)
            if gradient is None:
                print "\tGradient: [disabled]"
            else:
                print "\tGradient:",
                for d in range(self.nDim):
                    print gradient[d],
                print ""
        return result

    cpdef void bounceMove(self, double[:] x0, double[:] displacement, bint[:] bounced):
        '''
        x0: the initial position -- length nDim, has new position written to it
        displacement: How much we would like to move x by -- length nDim
        bounced: Output only - reports whether we bounced in each dimension -- length nDim
        '''
        cdef Size d, lastIndex
        cdef double boundary
        self.interp(x0)
        for d in range(self.nDim):
            lastIndex = self.indices[d]
            x0[d] += displacement[d]
            if(isnan(self.interp(x0))):
                # Find direction and boundary
                if displacement[d] > 0:
                    while self.indices[d] > lastIndex:
                        if not isnan(self.interp(x0,None,locate=False)):
                            break
                        self.indices[d] -= 1
                    boundary = self.findEdge(self.indices[d]+1,d)
                    x0[d] = 2*boundary - x0[d]
                else:
                    while self.indices[d] < lastIndex:
                        if not isnan(self.interp(x0,None,locate=False)):
                            break
                        self.indices[d] += 1
                    boundary = self.findEdge(self.indices[d],d)
                    x0[d] = 2*boundary - x0[d]
                bounced[d] = True
            else:
                bounced[d] = False
        return

    cpdef double findEdge(self, Size index, Size dim):
        if isnan(self.axes[dim,1]):
            return index*self.axes[dim,2] + self.axes[dim,0]
        return self.axes[dim,index]

    cpdef void interpN(self,double[:,:] points, double[:] output):
        cdef Size i
        assert output.shape[0] == points.shape[0]
        for i in range(points.shape[0]):
            output[i] = self.interp(points[i,:])
        return

    cpdef void printInfo(self):
        cdef Size d, i
        for d in range(self.nDim):
            print "Axis {0}:".format(d),
            if isnan(self.axes[d,1]):
                print "{0} ({1}:{2})".format(self.nPoints[d],self.axes[d,0],self.axes[d,3])
            else:
                print "{0} ({1}:{2})".format(self.nPoints[d],self.axes[d,0],self.axes[d,self.nPoints[d]-1])
        print "Strides:", np.asarray(self.strides)
        return
