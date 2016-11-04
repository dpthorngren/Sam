# distutils: language = c++
# distutils: sources = src/sam.cpp

import numpy as np
include "distributions.pyx"

# Special function wrappers
cpdef double incBeta(double x, double a, double b):
    return _incBeta(a,b,x)

cdef class Sam:
    cpdef void setRecordOptions(self,bint recordSamples, bint accumulateStats, bint printSamples):
        self.sam[0].setRecordOptions(recordSamples, accumulateStats, printSamples)
        return
