import numpy as np
from scipy.interpolate.interpnd import _ndim_coords_from_arrays

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
            if d > 0:
                strides[:d] *= self.nPoints[self.nDim-d]
            # Check for uniformity
            if np.allclose(axes[d],
                           np.linspace(axes[d][0], axes[d][-1], axes[d].size),
                           rtol=1e-6):
                uniform[d] = True
            else:
                uniform[d] = False
                if maxPoints < self.nPoints[d]:
                    maxPoints = self.nPoints[d]

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

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.cdivision(True)
    # @cython.initializedcheck(False)
    cdef Size ind(self,Size[:] p):
        cdef Size d, index = 0
        for d in range(self.nDim):
            index += p[d]*self.strides[d]
        return index

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.cdivision(True)
    # @cython.initializedcheck(False)
    cdef bint locatePoints(self, double[:] point):
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

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.cdivision(True)
    # @cython.initializedcheck(False)
    def __call__(self,x,gradient=False):
        # Using the RegularGridInterpolator system for sanity
        cdef Size i, nInputs
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
                outputsView[i] = self.interp(inputs[i],gradsView[i])
            if nInputs == 1:
                return outputs[0], grads[0]
            return outputs, grads
        else:
            for i in range(nInputs):
                outputsView[i] = self.interp(inputs[i])
            if nInputs==1:
                return outputs[0]
            return outputs

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.cdivision(True)
    # @cython.initializedcheck(False)
    cdef double interp(self,double[:] points, double [:] gradient=None, bint locate=True):
        # Locate indices and compute weights, or return nan
        if locate:
            if(self.locatePoints(points)):
                return nan
        cdef Size b, d, offset
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
            result += adjustment
            if gradient is not None:
                for d in range(self.nDim):
                    offset = b>>d&1
                    if offset:
                        gradient[d] += adjustment / (self.weights[d]*self.widths[d])
                    else:
                        gradient[d] -= adjustment / ((1-self.weights[d])*self.widths[d])
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

    cdef double findEdge(self, Size index, Size dim):
        if isnan(self.axes[dim,1]):
            return index*self.axes[dim,2] + self.axes[dim,0]
        return self.axes[dim,index]

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.cdivision(True)
    # @cython.initializedcheck(False)
    cdef void interpN(self,double[:,:] points, double[:] output):
        cdef Size i
        assert output.shape[0] == points.shape[0]
        for i in range(points.shape[0]):
            output[i] = self.interp(points[i,:])
        return


def _testF(x,y):
    return np.cos(x) + 2*y

def _testGradF(x,y):
    return np.array([-np.sin(x),2])


def testGriddy():
    x = (np.linspace(0,10,1000),
         np.sin(np.linspace(0,np.pi/2,900)))
    y = _testF(x[0][:,np.newaxis],x[1][np.newaxis,:])
    a = Griddy(x,y)
    print ""
    print "Call test:"
    print a((np.linspace(0,5,10),np.pi/4),gradient=True)
    print a((3.3,np.pi/4))
    b = a((2.3,np.pi/6.4),gradient=True)
    print b[0], b[1][0], b[1][1]
    print ""
    print "Dimension, Strides:"
    print a.nPoints[0], a.nPoints[1], ",", a.strides[0], a.strides[1]
    print ""
    print "Index Interpretation:"
    print a.ind(np.array([0,0],dtype=int)), len(a.values)
    print a.ind(np.array([10,4],dtype=int)), len(a.values)
    print ""
    print "Point Identification:"
    print a.locatePoints(np.array([5,np.pi/4],dtype=np.double)), a.indices[0], a.indices[1], a.weights[0], a.weights[1]
    print a.locatePoints(np.array([1,np.pi/8],dtype=np.double)), a.indices[0], a.indices[1], a.weights[0], a.weights[1]
    print a.locatePoints(np.array([10,0],dtype=np.double)), a.indices[0], a.indices[1], a.weights[0], a.weights[1]
    print a.locatePoints(np.array([0,np.pi/2],dtype=np.double)), a.indices[0], a.indices[1], a.weights[0], a.weights[1]
    print ""
    print "Grid value check:"
    print a.values[a.ind(np.array([50,33]))], _testF(x[0][50],np.sin(x[1][33]))
    print ""
    print "Interpolation Test:"
    print a.interp(np.array([5,np.pi/4],dtype=np.double)), _testF(5,np.pi/4)
    print a.interp(np.array([1,np.pi/8],dtype=np.double)), _testF(1,np.pi/8)
    print a.interp(np.array([-1,np.pi/8],dtype=np.double)), nan
    print ""
    print "Gradient Interpolation Test:"
    c = np.zeros(2)
    b = np.array([2.3,np.pi/6.4],dtype=np.double)
    a.interp(b,gradient=c)
    print c, _testGradF(b[0],b[1])
    b = np.array([5,np.pi/4],dtype=np.double)
    a.interp(b,gradient=c)
    print c, _testGradF(b[0],b[1])
    print ""
    print "Vectorized Interpolation Test:"
    b = np.array([[5,np.pi/4],[7.34,np.pi/6]],dtype=np.double)
    c = np.zeros(2)
    a.interpN(b,c)
    print c, _testF(5,np.pi/4), _testF(7.34,np.pi/6)
    print ""
    print "Test Complete."
    return
