# distutils: language = c++
# distutils: sources = src/sam.cpp

import numpy as np
include "distributions.pyx"

# Special function wrappers
cpdef double incBeta(double x, double a, double b):
    return _incBeta(a,b,x)

# Function storage routines.
cdef double _pyCallLogProb_(double* x, size_t nDim):
    return _pyLogProb_(np.asarray(<double[:nDim]>x))

cdef void registerCLogProb(string name, double(*func)(double*,size_t)):
    _cLogProb_[name] = func
    return

cdef void registerCustomSampler(string name, void (*func)(double*, double*, double*, size_t*, size_t, cRNG*, SubSamplerData*)):
    _customSamplerStore_[name] = func
    return

cdef void _pyCallCustom_(double* x, double* xPropose, double* working, size_t *nAccepted, size_t nDim, cRNG *rng, SubSamplerData *sub):
    return

cdef class PySam:
    cpdef void setRecordOptions(self,bint recordSamples, bint accumulateStats, bint printSamples):
        self.sam.setRecordOptions(recordSamples, accumulateStats, printSamples)
        return

    cpdef void write(self,string filename, bint writeHeader, string sep):
        self.sam.write(filename,writeHeader,sep)
        return

    cpdef void addMetropolis(self,np.ndarray proposalStd, size_t targetStart, size_t targetLen):
        cdef double[:] proposalView = proposalStd.copy()
        self.sam.addMetropolis(&proposalView[0],targetStart,targetLen)
        return

    cpdef void addCustom(self,object customFunc, object dData, object sData):
        cdef void (*customFuncPtr)(double*, double*, double*, size_t*, size_t, cRNG*, SubSamplerData*)
        cdef double[:] dDataView
        cdef size_t[:] sDataView
        cdef double* dDataPtr = NULL
        cdef size_t* sDataPtr = NULL
        cdef size_t dDataLen = 0
        cdef size_t sDataLen = 0
        cdef size_t newIndex
        # TODO: Enforce C ordering
        if dData is not None:
            dDataView = dData.copy()
            dDataPtr = &dDataView[0]
            dDataLen = dDataView.shape[0]
        if type(customFunc) is str:
            if _customSamplerStore_.count(customFunc) > 0:
                customFuncPtr = _customSamplerStore_[customFunc]
            else:
                raise ValueError("{} not found in function store."%customFunc)
            if sData is not None:
                sDataView = sData.astype(np.uintp)
                sDataPtr = &sDataView[0]
                sDataLen = sDataView.shape[0]
            self.sam.addCustom(customFuncPtr,dDataPtr,dDataLen,sDataPtr,sDataLen)
            return
        elif callable(customFunc):
            newIndex = len(_pyCustomSamps_)
            _pyCustomSamps_.append(customFunc)
            if sData is not None:
                sDataAugmented = np.empty(sData.shape[0]+1,dtype=np.uintp)
                sDataAugmented[1:] = sData
            else:
                sDataAugmented = np.array([1],dtype=np.uintp)
            sDataAugmented[0] = newIndex
            sDataView = sDataAugmented
            sDataPtr = &sDataView[0]
            sDataLen = sDataView.shape[0]
            self.sam.addCustom(_pyCallCustom_,dDataPtr,dDataLen,sDataPtr,sDataLen)
            return
        raise ValueError("First argument must be a callable or the ID of a registered function.")

    cpdef object getSamples(self):
        if self.sam.getSamples() == NULL:
            return np.empty((0,self.sam.getNDim()))
        return np.asarray(<double[:self.sam.getNSamples(),:self.sam.getNDim()]>self.sam.getSamples())

    cpdef object run(self, size_t nSamples, double[:] x0, size_t burnIn, size_t thin):
        if(x0.shape[0] != self.sam.getNDim()):
            raise ValueError("x0 has size {}, but nDim is {}".format(x0.shape[0],self.sam.getNDim()))
        self.sam.run(nSamples,&x0[0],burnIn,thin)
        return self.getSamples()

    def __init__(self, int nDim, object logProbability):
        self.sam = NULL
        cdef double (*logProbPtr)(double*, size_t)
        if type(logProbability) is str:
            if _cLogProb_.count(logProbability) > 0:
                logProbPtr = _cLogProb_[logProbability]
            else:
                raise ValueError("{} not found in function store."%logProbability)
        elif callable(logProbability):
            _pyLogProb_ = logProbability
            logProbPtr = _pyCallLogProb_
        else:
            raise TypeError("logProbability must be the ID of a registered c function or a callable object.")
        self.sam = new Sam(nDim, NULL)
        return

    def __dealloc__(self):
        if self.sam != NULL:
            del self.sam
            self.sam = NULL
        return

    def __str__(self):
        print self.sam.getStatus()
