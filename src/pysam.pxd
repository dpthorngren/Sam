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

# Wrapper for the RNG class
include "distributions.pxd"

cdef extern from "sam.h":
    cdef cppclass CppSam "Sam":
        Sam(size_t, double (*)(double*))
        Sam()
        void setRecordOptions(bint, bint, bint)
        void run(size_t, double*, size_t, size_t)
        double* getSamples()
        # void write(std::string, bool, std::string)
        void addMetropolis(double*, size_t, size_t)
        double getMean(size_t)
        double getVar(size_t)
        double getStd(size_t)

cdef class Sam:
    cdef CppSam *sam
    cpdef void setRecordOptions(self,bint, bint, bint)

