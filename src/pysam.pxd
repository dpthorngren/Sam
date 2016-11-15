from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.pair cimport pair
cimport numpy as np
import numpy as np

# Boost special functions
cdef extern from "<boost/math/special_functions.hpp>" namespace "boost::math":
    cpdef double asinh(double x) except +
    cpdef double acosh(double x) except +
    cpdef double atanh(double x) except +
    cpdef double beta(double a, double b) except +
    cdef double _incBeta "boost::math::beta"(double a, double b, double x) except +
    cpdef double gamma "boost::math::tgamma"(double x) except +
    cpdef double digamma(double x) except +
    cpdef double binomial_coefficient[double](unsigned int n, unsigned int k) except +

# Wrapper to fix ordering not matching other functions.
cpdef double incBeta(double x, double a, double b)

# Wrapper for the cRNG class (auto-generated)
include "distributions.pxd"

cdef extern from "sam.h":
    struct SubSamplerData:
        pass
    cdef cppclass Sam:
        Sam(size_t, double (*)(double*))
        Sam()
        void setRecordOptions(bint, bint, bint)
        void run(size_t, double*, size_t, size_t)
        string getStatus();
        double* getSamples()
        void write(string, bool, string)
        void addMetropolis(double*, size_t, size_t)
        void addCustom(void (*)(double*, double*, double*, size_t*, size_t, cRNG*, SubSamplerData*), double*, size_t, size_t*, size_t)
        double getMean(size_t)
        double getVar(size_t)
        double getStd(size_t)
        size_t getNDim()
        size_t getNSamples()

# Manage log probability functions
cdef object _pyLogProb_
cdef double _pyCallLogProb_(double*,size_t)
cdef map[string,double(*)(double*,size_t)] _cLogProb_
cdef void registerCLogProb(string, double(*)(double*,size_t))
# Manage custom sampling functions
cdef object _pyCustomSamps_ = []
cdef void _pyCallCustom_(double*, double*, double*, size_t*, size_t, cRNG*, SubSamplerData*)
cdef map[string,void (*)(double*, double*, double*, size_t*, size_t, cRNG*, SubSamplerData*)] _customSamplerStore_
cdef void registerCustomSampler(string,void (*)(double*, double*, double*, size_t*, size_t, cRNG*, SubSamplerData*))

cdef class PySam:
    cdef Sam *sam
    cpdef void setRecordOptions(self,bint, bint, bint)
    cpdef void write(self,string, bint, string)
    cpdef void addMetropolis(self,np.ndarray proposalStd, size_t targetStart, size_t targetLen)
    cpdef void addCustom(self,object customFunc, object dData, object sData)
    cpdef object getSamples(self)
    cpdef object run(self, size_t, double[:], size_t, size_t)
