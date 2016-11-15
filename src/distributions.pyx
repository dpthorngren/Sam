
cdef class RNG:
    cdef double _normalMean(self, double mean, double std):
        return self.rng.normalMean(mean, std)
    cpdef object normalMean(self, object mean, object std):
        return self.wrapDDD(RNG._normalMean,mean, std)
    cdef double _normalVar(self, double mean, double std):
        return self.rng.normalVar(mean, std)
    cpdef object normalVar(self, object mean, object std):
        return self.wrapDDD(RNG._normalVar,mean, std)
    cdef double _normalStd(self, double mean, double std):
        return self.rng.normalStd(mean, std)
    cpdef object normalStd(self, object mean, object std):
        return self.wrapDDD(RNG._normalStd,mean, std)
    cdef double _normalRand(self, double mean, double std):
        return self.rng.normalRand(mean, std)
    cpdef object normalRand(self, object mean, object std):
        return self.wrapDDD(RNG._normalRand,mean, std)
    cdef double _normalPDF(self, double x, double mean, double std):
        return self.rng.normalPDF(x, mean, std)
    cpdef object normalPDF(self, object x, object mean, object std):
        return self.wrapDDDD(RNG._normalPDF,x, mean, std)
    cdef double _normalLogPDF(self, double x, double mean, double std):
        return self.rng.normalLogPDF(x, mean, std)
    cpdef object normalLogPDF(self, object x, object mean, object std):
        return self.wrapDDDD(RNG._normalLogPDF,x, mean, std)
    cdef double _uniformMean(self, double xMin, double xMax):
        return self.rng.uniformMean(xMin, xMax)
    cpdef object uniformMean(self, object xMin, object xMax):
        return self.wrapDDD(RNG._uniformMean,xMin, xMax)
    cdef double _uniformVar(self, double xMin, double xMax):
        return self.rng.uniformVar(xMin, xMax)
    cpdef object uniformVar(self, object xMin, object xMax):
        return self.wrapDDD(RNG._uniformVar,xMin, xMax)
    cdef double _uniformStd(self, double xMin, double xMax):
        return self.rng.uniformStd(xMin, xMax)
    cpdef object uniformStd(self, object xMin, object xMax):
        return self.wrapDDD(RNG._uniformStd,xMin, xMax)
    cdef double _uniformRand(self, double xMin, double xMax):
        return self.rng.uniformRand(xMin, xMax)
    cpdef object uniformRand(self, object xMin, object xMax):
        return self.wrapDDD(RNG._uniformRand,xMin, xMax)
    cdef double _uniformPDF(self, double x, double xMin, double xMax):
        return self.rng.uniformPDF(x, xMin, xMax)
    cpdef object uniformPDF(self, object x, object xMin, object xMax):
        return self.wrapDDDD(RNG._uniformPDF,x, xMin, xMax)
    cdef double _uniformLogPDF(self, double x, double xMin, double xMax):
        return self.rng.uniformLogPDF(x, xMin, xMax)
    cpdef object uniformLogPDF(self, object x, object xMin, object xMax):
        return self.wrapDDDD(RNG._uniformLogPDF,x, xMin, xMax)
    cdef double _uniformCDF(self, double x, double xMin, double xMax):
        return self.rng.uniformCDF(x, xMin, xMax)
    cpdef object uniformCDF(self, object x, object xMin, object xMax):
        return self.wrapDDDD(RNG._uniformCDF,x, xMin, xMax)
    cdef double _uniformIntMean(self, int xMin, int xMax):
        return self.rng.uniformIntMean(xMin, xMax)
    cpdef object uniformIntMean(self, object xMin, object xMax):
        return self.wrapDII(RNG._uniformIntMean,xMin, xMax)
    cdef double _uniformIntVar(self, int xMin, int xMax):
        return self.rng.uniformIntVar(xMin, xMax)
    cpdef object uniformIntVar(self, object xMin, object xMax):
        return self.wrapDII(RNG._uniformIntVar,xMin, xMax)
    cdef double _uniformIntStd(self, int xMin, int xMax):
        return self.rng.uniformIntStd(xMin, xMax)
    cpdef object uniformIntStd(self, object xMin, object xMax):
        return self.wrapDII(RNG._uniformIntStd,xMin, xMax)
    cdef int _uniformIntRand(self, int xMin, int xMax):
        return self.rng.uniformIntRand(xMin, xMax)
    cpdef object uniformIntRand(self, object xMin, object xMax):
        return self.wrapIII(RNG._uniformIntRand,xMin, xMax)
    cdef double _uniformIntPDF(self, int x, int xMin, int xMax):
        return self.rng.uniformIntPDF(x, xMin, xMax)
    cpdef object uniformIntPDF(self, object x, object xMin, object xMax):
        return self.wrapDIII(RNG._uniformIntPDF,x, xMin, xMax)
    cdef double _uniformIntLogPDF(self, int x, int xMin, int xMax):
        return self.rng.uniformIntLogPDF(x, xMin, xMax)
    cpdef object uniformIntLogPDF(self, object x, object xMin, object xMax):
        return self.wrapDIII(RNG._uniformIntLogPDF,x, xMin, xMax)
    cdef double _uniformIntCDF(self, double x, int xMin, int xMax):
        return self.rng.uniformIntCDF(x, xMin, xMax)
    cpdef object uniformIntCDF(self, object x, object xMin, object xMax):
        return self.wrapDDII(RNG._uniformIntCDF,x, xMin, xMax)
    cdef double _gammaMean(self, double shape, double rate):
        return self.rng.gammaMean(shape, rate)
    cpdef object gammaMean(self, object shape, object rate):
        return self.wrapDDD(RNG._gammaMean,shape, rate)
    cdef double _gammaVar(self, double shape, double rate):
        return self.rng.gammaVar(shape, rate)
    cpdef object gammaVar(self, object shape, object rate):
        return self.wrapDDD(RNG._gammaVar,shape, rate)
    cdef double _gammaStd(self, double shape, double rate):
        return self.rng.gammaStd(shape, rate)
    cpdef object gammaStd(self, object shape, object rate):
        return self.wrapDDD(RNG._gammaStd,shape, rate)
    cdef double _gammaRand(self, double shape, double rate):
        return self.rng.gammaRand(shape, rate)
    cpdef object gammaRand(self, object shape, object rate):
        return self.wrapDDD(RNG._gammaRand,shape, rate)
    cdef double _gammaPDF(self, double x, double shape, double rate):
        return self.rng.gammaPDF(x, shape, rate)
    cpdef object gammaPDF(self, object x, object shape, object rate):
        return self.wrapDDDD(RNG._gammaPDF,x, shape, rate)
    cdef double _gammaLogPDF(self, double x, double shape, double rate):
        return self.rng.gammaLogPDF(x, shape, rate)
    cpdef object gammaLogPDF(self, object x, object shape, object rate):
        return self.wrapDDDD(RNG._gammaLogPDF,x, shape, rate)
    cdef double _gammaCDF(self, double x, double shape, double rate):
        return self.rng.gammaCDF(x, shape, rate)
    cpdef object gammaCDF(self, object x, object shape, object rate):
        return self.wrapDDDD(RNG._gammaCDF,x, shape, rate)
    cdef double _invGammaMean(self, double shape, double rate):
        return self.rng.invGammaMean(shape, rate)
    cpdef object invGammaMean(self, object shape, object rate):
        return self.wrapDDD(RNG._invGammaMean,shape, rate)
    cdef double _invGammaVar(self, double shape, double rate):
        return self.rng.invGammaVar(shape, rate)
    cpdef object invGammaVar(self, object shape, object rate):
        return self.wrapDDD(RNG._invGammaVar,shape, rate)
    cdef double _invGammaStd(self, double shape, double rate):
        return self.rng.invGammaStd(shape, rate)
    cpdef object invGammaStd(self, object shape, object rate):
        return self.wrapDDD(RNG._invGammaStd,shape, rate)
    cdef double _invGammaRand(self, double shape, double rate):
        return self.rng.invGammaRand(shape, rate)
    cpdef object invGammaRand(self, object shape, object rate):
        return self.wrapDDD(RNG._invGammaRand,shape, rate)
    cdef double _invGammaPDF(self, double x, double shape, double rate):
        return self.rng.invGammaPDF(x, shape, rate)
    cpdef object invGammaPDF(self, object x, object shape, object rate):
        return self.wrapDDDD(RNG._invGammaPDF,x, shape, rate)
    cdef double _invGammaLogPDF(self, double x, double shape, double rate):
        return self.rng.invGammaLogPDF(x, shape, rate)
    cpdef object invGammaLogPDF(self, object x, object shape, object rate):
        return self.wrapDDDD(RNG._invGammaLogPDF,x, shape, rate)
    cdef double _invGammaCDF(self, double x, double shape, double rate):
        return self.rng.invGammaCDF(x, shape, rate)
    cpdef object invGammaCDF(self, object x, object shape, object rate):
        return self.wrapDDDD(RNG._invGammaCDF,x, shape, rate)
    cdef double _betaMean(self, double alpha, double beta):
        return self.rng.betaMean(alpha, beta)
    cpdef object betaMean(self, object alpha, object beta):
        return self.wrapDDD(RNG._betaMean,alpha, beta)
    cdef double _betaVar(self, double alpha, double beta):
        return self.rng.betaVar(alpha, beta)
    cpdef object betaVar(self, object alpha, object beta):
        return self.wrapDDD(RNG._betaVar,alpha, beta)
    cdef double _betaStd(self, double alpha, double beta):
        return self.rng.betaStd(alpha, beta)
    cpdef object betaStd(self, object alpha, object beta):
        return self.wrapDDD(RNG._betaStd,alpha, beta)
    cdef double _betaRand(self, double alpha, double beta):
        return self.rng.betaRand(alpha, beta)
    cpdef object betaRand(self, object alpha, object beta):
        return self.wrapDDD(RNG._betaRand,alpha, beta)
    cdef double _betaPDF(self, double x, double alpha, double beta):
        return self.rng.betaPDF(x, alpha, beta)
    cpdef object betaPDF(self, object x, object alpha, object beta):
        return self.wrapDDDD(RNG._betaPDF,x, alpha, beta)
    cdef double _betaLogPDF(self, double x, double alpha, double beta):
        return self.rng.betaLogPDF(x, alpha, beta)
    cpdef object betaLogPDF(self, object x, object alpha, object beta):
        return self.wrapDDDD(RNG._betaLogPDF,x, alpha, beta)
    cdef double _betaCDF(self, double x, double alpha, double beta):
        return self.rng.betaCDF(x, alpha, beta)
    cpdef object betaCDF(self, object x, object alpha, object beta):
        return self.wrapDDDD(RNG._betaCDF,x, alpha, beta)
    cdef double _poissonMean(self, double rate):
        return self.rng.poissonMean(rate)
    cpdef object poissonMean(self, object rate):
        return self.wrapDD(RNG._poissonMean,rate)
    cdef double _poissonVar(self, double rate):
        return self.rng.poissonVar(rate)
    cpdef object poissonVar(self, object rate):
        return self.wrapDD(RNG._poissonVar,rate)
    cdef double _poissonStd(self, double rate):
        return self.rng.poissonStd(rate)
    cpdef object poissonStd(self, object rate):
        return self.wrapDD(RNG._poissonStd,rate)
    cdef int _poissonRand(self, double rate):
        return self.rng.poissonRand(rate)
    cpdef object poissonRand(self, object rate):
        return self.wrapID(RNG._poissonRand,rate)
    cdef double _poissonPDF(self, int x, double rate):
        return self.rng.poissonPDF(x, rate)
    cpdef object poissonPDF(self, object x, object rate):
        return self.wrapDID(RNG._poissonPDF,x, rate)
    cdef double _poissonLogPDF(self, int x, double rate):
        return self.rng.poissonLogPDF(x, rate)
    cpdef object poissonLogPDF(self, object x, object rate):
        return self.wrapDID(RNG._poissonLogPDF,x, rate)
    cdef double _poissonCDF(self, double x, double rate):
        return self.rng.poissonCDF(x, rate)
    cpdef object poissonCDF(self, object x, object rate):
        return self.wrapDDD(RNG._poissonCDF,x, rate)
    cdef double _exponentialMean(self, double rate):
        return self.rng.exponentialMean(rate)
    cpdef object exponentialMean(self, object rate):
        return self.wrapDD(RNG._exponentialMean,rate)
    cdef double _exponentialVar(self, double rate):
        return self.rng.exponentialVar(rate)
    cpdef object exponentialVar(self, object rate):
        return self.wrapDD(RNG._exponentialVar,rate)
    cdef double _exponentialStd(self, double rate):
        return self.rng.exponentialStd(rate)
    cpdef object exponentialStd(self, object rate):
        return self.wrapDD(RNG._exponentialStd,rate)
    cdef double _exponentialRand(self, double rate):
        return self.rng.exponentialRand(rate)
    cpdef object exponentialRand(self, object rate):
        return self.wrapDD(RNG._exponentialRand,rate)
    cdef double _exponentialPDF(self, double x, double rate):
        return self.rng.exponentialPDF(x, rate)
    cpdef object exponentialPDF(self, object x, object rate):
        return self.wrapDDD(RNG._exponentialPDF,x, rate)
    cdef double _exponentialLogPDF(self, double x, double rate):
        return self.rng.exponentialLogPDF(x, rate)
    cpdef object exponentialLogPDF(self, object x, object rate):
        return self.wrapDDD(RNG._exponentialLogPDF,x, rate)
    cdef double _exponentialCDF(self, double x, double rate):
        return self.rng.exponentialCDF(x, rate)
    cpdef object exponentialCDF(self, object x, object rate):
        return self.wrapDDD(RNG._exponentialCDF,x, rate)
    cdef double _binomialMean(self, int number, double probability):
        return self.rng.binomialMean(number, probability)
    cpdef object binomialMean(self, object number, object probability):
        return self.wrapDID(RNG._binomialMean,number, probability)
    cdef double _binomialVar(self, int number, double probability):
        return self.rng.binomialVar(number, probability)
    cpdef object binomialVar(self, object number, object probability):
        return self.wrapDID(RNG._binomialVar,number, probability)
    cdef double _binomialStd(self, int number, double probability):
        return self.rng.binomialStd(number, probability)
    cpdef object binomialStd(self, object number, object probability):
        return self.wrapDID(RNG._binomialStd,number, probability)
    cdef int _binomialRand(self, int number, double probability):
        return self.rng.binomialRand(number, probability)
    cpdef object binomialRand(self, object number, object probability):
        return self.wrapIID(RNG._binomialRand,number, probability)
    cdef double _binomialPDF(self, int x, int number, double probability):
        return self.rng.binomialPDF(x, number, probability)
    cpdef object binomialPDF(self, object x, object number, object probability):
        return self.wrapDIID(RNG._binomialPDF,x, number, probability)
    cdef double _binomialLogPDF(self, int x, int number, double probability):
        return self.rng.binomialLogPDF(x, number, probability)
    cpdef object binomialLogPDF(self, object x, object number, object probability):
        return self.wrapDIID(RNG._binomialLogPDF,x, number, probability)
    cdef double _binomialCDF(self, double x, int number, double probability):
        return self.rng.binomialCDF(x, number, probability)
    cpdef object binomialCDF(self, object x, object number, object probability):
        return self.wrapDDID(RNG._binomialCDF,x, number, probability)
    cdef object wrapDD(self, double (*func)(RNG, double), object arg1):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef double[:] view
        cdef double[:] output
        if type(arg1) is np.ndarray:
            view = arg1.ravel()
            outputObj = np.empty_like(arg1)
            output = outputObj.ravel()
            for i in range(view.shape[0]):
                output[i] = func(self,view[i])
            return outputObj
        return func(self,<double>arg1)
    cdef object wrapDDD(self, double (*func)(RNG, double, double), object arg1, object arg2):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef double[:] view1
        cdef double[:] view2
        cdef double[:] output
        cdef double dArg1
        cdef double dArg2
        cdef type tArg1 = type(arg1), tArg2 = type(arg2)
        if tArg1 is np.ndarray and tArg2 is np.ndarray:
            if arg1.shape != arg2.shape:
                raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
            view1 = arg1.ravel()
            view2 = arg2.ravel()
            outputObj = np.empty_like(arg1)
            output = outputObj.ravel()
            for i in range(view1.shape[0]):
                output[i] = func(self,view1[i],view2[i])
            return outputObj
        if tArg1 is np.ndarray and tArg2 is not np.ndarray:
            view1 = arg1.ravel()
            outputObj = np.empty_like(arg1)
            output = outputObj.ravel()
            dArg2 = arg2
            for i in range(view1.shape[0]):
                output[i] = func(self,view1[i],dArg2)
            return outputObj
        if tArg2 is np.ndarray and tArg1 is not np.ndarray:
            view2 = arg2.ravel()
            outputObj = np.empty_like(arg2)
            output = outputObj.ravel()
            dArg1 = arg1
            for i in range(view2.shape[0]):
                output[i] = func(self,dArg1,view2[i])
            return outputObj
        dArg1, dArg2 = arg1, arg2
        return func(self,dArg1,dArg2)
    cdef object wrapDDDD(self, double (*func)(RNG, double, double, double), object arg1, object arg2, object arg3):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef double[:] view1
        cdef double cArg1
        cdef double[:] view2
        cdef double cArg2
        cdef double[:] view3
        cdef double cArg3
        cdef double[:] output
        cdef type tArg1 = type(arg1)
        cdef type tArg2 = type(arg2)
        cdef type tArg3 = type(arg3)
        if tArg1 is np.ndarray:
            if tArg2 is np.ndarray:
                if tArg3 is np.ndarray: # array, array, array
                    view1 = arg1.ravel()
                    view2 = arg2.ravel()
                    view3 = arg3.ravel()
                    if arg1.shape != arg2.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
                    if arg2.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg2.shape,arg3.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],view2[i],view3[i])
                    return outputObj
                else: # array, array, double
                    view1 = arg1.ravel()
                    view2 = arg2.ravel()
                    cArg3 = arg3
                    if arg1.shape != arg2.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],view2[i],cArg3)
                    return outputObj
            else:
                if tArg3 is np.ndarray: # array, double, array
                    view1 = arg1.ravel()
                    cArg2 = arg2
                    view3 = arg3.ravel()
                    if arg1.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg3.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],cArg2,view3[i])
                    return outputObj
                else: # array, double, double
                    view1 = arg1.ravel()
                    cArg2, cArg3 = arg2, arg3
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],cArg2,cArg3)
                    return outputObj
        else:
            if tArg2 is np.ndarray:
                if tArg3 is np.ndarray: # double, array, array
                    cArg1 = arg1
                    view2 = arg2.ravel()
                    view3 = arg3.ravel()
                    if arg2.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg2.shape,arg3.shape))
                    outputObj = np.empty_like(arg2)
                    output = outputObj.ravel()
                    for i in range(view2.shape[0]):
                        output[i] = func(self,cArg1,view2[i],view3[i])
                    return outputObj
                else: # double, array, double
                    cArg1, cArg3 = arg1, arg3
                    view2 = arg2.ravel()
                    outputObj = np.empty_like(arg2)
                    output = outputObj.ravel()
                    for i in range(view2.shape[0]):
                        output[i] = func(self,cArg1,view2[i],cArg3)
                    return outputObj
            else:
                if tArg3 is np.ndarray: # double, double, array
                    cArg1, cArg2 = arg1, arg2
                    view3 = arg3.ravel()
                    outputObj = np.empty_like(arg3)
                    output = outputObj.ravel()
                    for i in range(view3.shape[0]):
                        output[i] = func(self,cArg1,cArg2,view3[i])
                    return outputObj
                else: # double, double, double
                    cArg1, cArg2, cArg3 = arg1, arg2, arg3
                    return func(self,cArg1,cArg2,cArg3)
    cdef object wrapDDID(self, double (*func)(RNG, double, int, double), object arg1, object arg2, object arg3):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef double[:] view1
        cdef double cArg1
        cdef int[:] view2
        cdef int cArg2
        cdef double[:] view3
        cdef double cArg3
        cdef double[:] output
        cdef type tArg1 = type(arg1)
        cdef type tArg2 = type(arg2)
        cdef type tArg3 = type(arg3)
        if tArg1 is np.ndarray:
            if tArg2 is np.ndarray:
                if tArg3 is np.ndarray: # array, array, array
                    view1 = arg1.ravel()
                    view2 = arg2.ravel()
                    view3 = arg3.ravel()
                    if arg1.shape != arg2.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
                    if arg2.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg2.shape,arg3.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],view2[i],view3[i])
                    return outputObj
                else: # array, array, double
                    view1 = arg1.ravel()
                    view2 = arg2.ravel()
                    cArg3 = arg3
                    if arg1.shape != arg2.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],view2[i],cArg3)
                    return outputObj
            else:
                if tArg3 is np.ndarray: # array, int, array
                    view1 = arg1.ravel()
                    cArg2 = arg2
                    view3 = arg3.ravel()
                    if arg1.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg3.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],cArg2,view3[i])
                    return outputObj
                else: # array, int, double
                    view1 = arg1.ravel()
                    cArg2, cArg3 = arg2, arg3
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],cArg2,cArg3)
                    return outputObj
        else:
            if tArg2 is np.ndarray:
                if tArg3 is np.ndarray: # double, array, array
                    cArg1 = arg1
                    view2 = arg2.ravel()
                    view3 = arg3.ravel()
                    if arg2.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg2.shape,arg3.shape))
                    outputObj = np.empty_like(arg2)
                    output = outputObj.ravel()
                    for i in range(view2.shape[0]):
                        output[i] = func(self,cArg1,view2[i],view3[i])
                    return outputObj
                else: # double, array, double
                    cArg1, cArg3 = arg1, arg3
                    view2 = arg2.ravel()
                    outputObj = np.empty_like(arg2)
                    output = outputObj.ravel()
                    for i in range(view2.shape[0]):
                        output[i] = func(self,cArg1,view2[i],cArg3)
                    return outputObj
            else:
                if tArg3 is np.ndarray: # double, int, array
                    cArg1, cArg2 = arg1, arg2
                    view3 = arg3.ravel()
                    outputObj = np.empty_like(arg3)
                    output = outputObj.ravel()
                    for i in range(view3.shape[0]):
                        output[i] = func(self,cArg1,cArg2,view3[i])
                    return outputObj
                else: # double, int, double
                    cArg1, cArg2, cArg3 = arg1, arg2, arg3
                    return func(self,cArg1,cArg2,cArg3)
    cdef object wrapDDII(self, double (*func)(RNG, double, int, int), object arg1, object arg2, object arg3):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef double[:] view1
        cdef double cArg1
        cdef int[:] view2
        cdef int cArg2
        cdef int[:] view3
        cdef int cArg3
        cdef double[:] output
        cdef type tArg1 = type(arg1)
        cdef type tArg2 = type(arg2)
        cdef type tArg3 = type(arg3)
        if tArg1 is np.ndarray:
            if tArg2 is np.ndarray:
                if tArg3 is np.ndarray: # array, array, array
                    view1 = arg1.ravel()
                    view2 = arg2.ravel()
                    view3 = arg3.ravel()
                    if arg1.shape != arg2.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
                    if arg2.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg2.shape,arg3.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],view2[i],view3[i])
                    return outputObj
                else: # array, array, int
                    view1 = arg1.ravel()
                    view2 = arg2.ravel()
                    cArg3 = arg3
                    if arg1.shape != arg2.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],view2[i],cArg3)
                    return outputObj
            else:
                if tArg3 is np.ndarray: # array, int, array
                    view1 = arg1.ravel()
                    cArg2 = arg2
                    view3 = arg3.ravel()
                    if arg1.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg3.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],cArg2,view3[i])
                    return outputObj
                else: # array, int, int
                    view1 = arg1.ravel()
                    cArg2, cArg3 = arg2, arg3
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],cArg2,cArg3)
                    return outputObj
        else:
            if tArg2 is np.ndarray:
                if tArg3 is np.ndarray: # double, array, array
                    cArg1 = arg1
                    view2 = arg2.ravel()
                    view3 = arg3.ravel()
                    if arg2.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg2.shape,arg3.shape))
                    outputObj = np.empty_like(arg2)
                    output = outputObj.ravel()
                    for i in range(view2.shape[0]):
                        output[i] = func(self,cArg1,view2[i],view3[i])
                    return outputObj
                else: # double, array, int
                    cArg1, cArg3 = arg1, arg3
                    view2 = arg2.ravel()
                    outputObj = np.empty_like(arg2)
                    output = outputObj.ravel()
                    for i in range(view2.shape[0]):
                        output[i] = func(self,cArg1,view2[i],cArg3)
                    return outputObj
            else:
                if tArg3 is np.ndarray: # double, int, array
                    cArg1, cArg2 = arg1, arg2
                    view3 = arg3.ravel()
                    outputObj = np.empty_like(arg3)
                    output = outputObj.ravel()
                    for i in range(view3.shape[0]):
                        output[i] = func(self,cArg1,cArg2,view3[i])
                    return outputObj
                else: # double, int, int
                    cArg1, cArg2, cArg3 = arg1, arg2, arg3
                    return func(self,cArg1,cArg2,cArg3)
    cdef object wrapDID(self, double (*func)(RNG, int, double), object arg1, object arg2):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef int[:] view1
        cdef double[:] view2
        cdef double[:] output
        cdef int dArg1
        cdef double dArg2
        cdef type tArg1 = type(arg1), tArg2 = type(arg2)
        if tArg1 is np.ndarray and tArg2 is np.ndarray:
            if arg1.shape != arg2.shape:
                raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
            view1 = arg1.ravel()
            view2 = arg2.ravel()
            outputObj = np.empty_like(arg1)
            output = outputObj.ravel()
            for i in range(view1.shape[0]):
                output[i] = func(self,view1[i],view2[i])
            return outputObj
        if tArg1 is np.ndarray and tArg2 is not np.ndarray:
            view1 = arg1.ravel()
            outputObj = np.empty_like(arg1)
            output = outputObj.ravel()
            dArg2 = arg2
            for i in range(view1.shape[0]):
                output[i] = func(self,view1[i],dArg2)
            return outputObj
        if tArg2 is np.ndarray and tArg1 is not np.ndarray:
            view2 = arg2.ravel()
            outputObj = np.empty_like(arg2)
            output = outputObj.ravel()
            dArg1 = arg1
            for i in range(view2.shape[0]):
                output[i] = func(self,dArg1,view2[i])
            return outputObj
        dArg1, dArg2 = arg1, arg2
        return func(self,dArg1,dArg2)
    cdef object wrapDII(self, double (*func)(RNG, int, int), object arg1, object arg2):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef int[:] view1
        cdef int[:] view2
        cdef double[:] output
        cdef int dArg1
        cdef int dArg2
        cdef type tArg1 = type(arg1), tArg2 = type(arg2)
        if tArg1 is np.ndarray and tArg2 is np.ndarray:
            if arg1.shape != arg2.shape:
                raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
            view1 = arg1.ravel()
            view2 = arg2.ravel()
            outputObj = np.empty_like(arg1)
            output = outputObj.ravel()
            for i in range(view1.shape[0]):
                output[i] = func(self,view1[i],view2[i])
            return outputObj
        if tArg1 is np.ndarray and tArg2 is not np.ndarray:
            view1 = arg1.ravel()
            outputObj = np.empty_like(arg1)
            output = outputObj.ravel()
            dArg2 = arg2
            for i in range(view1.shape[0]):
                output[i] = func(self,view1[i],dArg2)
            return outputObj
        if tArg2 is np.ndarray and tArg1 is not np.ndarray:
            view2 = arg2.ravel()
            outputObj = np.empty_like(arg2)
            output = outputObj.ravel()
            dArg1 = arg1
            for i in range(view2.shape[0]):
                output[i] = func(self,dArg1,view2[i])
            return outputObj
        dArg1, dArg2 = arg1, arg2
        return func(self,dArg1,dArg2)
    cdef object wrapDIID(self, double (*func)(RNG, int, int, double), object arg1, object arg2, object arg3):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef int[:] view1
        cdef int cArg1
        cdef int[:] view2
        cdef int cArg2
        cdef double[:] view3
        cdef double cArg3
        cdef double[:] output
        cdef type tArg1 = type(arg1)
        cdef type tArg2 = type(arg2)
        cdef type tArg3 = type(arg3)
        if tArg1 is np.ndarray:
            if tArg2 is np.ndarray:
                if tArg3 is np.ndarray: # array, array, array
                    view1 = arg1.ravel()
                    view2 = arg2.ravel()
                    view3 = arg3.ravel()
                    if arg1.shape != arg2.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
                    if arg2.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg2.shape,arg3.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],view2[i],view3[i])
                    return outputObj
                else: # array, array, double
                    view1 = arg1.ravel()
                    view2 = arg2.ravel()
                    cArg3 = arg3
                    if arg1.shape != arg2.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],view2[i],cArg3)
                    return outputObj
            else:
                if tArg3 is np.ndarray: # array, int, array
                    view1 = arg1.ravel()
                    cArg2 = arg2
                    view3 = arg3.ravel()
                    if arg1.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg3.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],cArg2,view3[i])
                    return outputObj
                else: # array, int, double
                    view1 = arg1.ravel()
                    cArg2, cArg3 = arg2, arg3
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],cArg2,cArg3)
                    return outputObj
        else:
            if tArg2 is np.ndarray:
                if tArg3 is np.ndarray: # int, array, array
                    cArg1 = arg1
                    view2 = arg2.ravel()
                    view3 = arg3.ravel()
                    if arg2.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg2.shape,arg3.shape))
                    outputObj = np.empty_like(arg2)
                    output = outputObj.ravel()
                    for i in range(view2.shape[0]):
                        output[i] = func(self,cArg1,view2[i],view3[i])
                    return outputObj
                else: # int, array, double
                    cArg1, cArg3 = arg1, arg3
                    view2 = arg2.ravel()
                    outputObj = np.empty_like(arg2)
                    output = outputObj.ravel()
                    for i in range(view2.shape[0]):
                        output[i] = func(self,cArg1,view2[i],cArg3)
                    return outputObj
            else:
                if tArg3 is np.ndarray: # int, int, array
                    cArg1, cArg2 = arg1, arg2
                    view3 = arg3.ravel()
                    outputObj = np.empty_like(arg3)
                    output = outputObj.ravel()
                    for i in range(view3.shape[0]):
                        output[i] = func(self,cArg1,cArg2,view3[i])
                    return outputObj
                else: # int, int, double
                    cArg1, cArg2, cArg3 = arg1, arg2, arg3
                    return func(self,cArg1,cArg2,cArg3)
    cdef object wrapDIII(self, double (*func)(RNG, int, int, int), object arg1, object arg2, object arg3):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef int[:] view1
        cdef int cArg1
        cdef int[:] view2
        cdef int cArg2
        cdef int[:] view3
        cdef int cArg3
        cdef double[:] output
        cdef type tArg1 = type(arg1)
        cdef type tArg2 = type(arg2)
        cdef type tArg3 = type(arg3)
        if tArg1 is np.ndarray:
            if tArg2 is np.ndarray:
                if tArg3 is np.ndarray: # array, array, array
                    view1 = arg1.ravel()
                    view2 = arg2.ravel()
                    view3 = arg3.ravel()
                    if arg1.shape != arg2.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
                    if arg2.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg2.shape,arg3.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],view2[i],view3[i])
                    return outputObj
                else: # array, array, int
                    view1 = arg1.ravel()
                    view2 = arg2.ravel()
                    cArg3 = arg3
                    if arg1.shape != arg2.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],view2[i],cArg3)
                    return outputObj
            else:
                if tArg3 is np.ndarray: # array, int, array
                    view1 = arg1.ravel()
                    cArg2 = arg2
                    view3 = arg3.ravel()
                    if arg1.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg3.shape))
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],cArg2,view3[i])
                    return outputObj
                else: # array, int, int
                    view1 = arg1.ravel()
                    cArg2, cArg3 = arg2, arg3
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],cArg2,cArg3)
                    return outputObj
        else:
            if tArg2 is np.ndarray:
                if tArg3 is np.ndarray: # int, array, array
                    cArg1 = arg1
                    view2 = arg2.ravel()
                    view3 = arg3.ravel()
                    if arg2.shape != arg3.shape:
                        raise ValueError("Operand shapes do not match: {}, {}".format(arg2.shape,arg3.shape))
                    outputObj = np.empty_like(arg2)
                    output = outputObj.ravel()
                    for i in range(view2.shape[0]):
                        output[i] = func(self,cArg1,view2[i],view3[i])
                    return outputObj
                else: # int, array, int
                    cArg1, cArg3 = arg1, arg3
                    view2 = arg2.ravel()
                    outputObj = np.empty_like(arg2)
                    output = outputObj.ravel()
                    for i in range(view2.shape[0]):
                        output[i] = func(self,cArg1,view2[i],cArg3)
                    return outputObj
            else:
                if tArg3 is np.ndarray: # int, int, array
                    cArg1, cArg2 = arg1, arg2
                    view3 = arg3.ravel()
                    outputObj = np.empty_like(arg3)
                    output = outputObj.ravel()
                    for i in range(view3.shape[0]):
                        output[i] = func(self,cArg1,cArg2,view3[i])
                    return outputObj
                else: # int, int, int
                    cArg1, cArg2, cArg3 = arg1, arg2, arg3
                    return func(self,cArg1,cArg2,cArg3)
    cdef object wrapID(self, int (*func)(RNG, double), object arg1):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef double[:] view
        cdef int[:] output
        if type(arg1) is np.ndarray:
            view = arg1.ravel()
            outputObj = np.empty_like(arg1)
            output = outputObj.ravel()
            for i in range(view.shape[0]):
                output[i] = func(self,view[i])
            return outputObj
        return func(self,<double>arg1)
    cdef object wrapIID(self, int (*func)(RNG, int, double), object arg1, object arg2):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef int[:] view1
        cdef double[:] view2
        cdef int[:] output
        cdef int dArg1
        cdef double dArg2
        cdef type tArg1 = type(arg1), tArg2 = type(arg2)
        if tArg1 is np.ndarray and tArg2 is np.ndarray:
            if arg1.shape != arg2.shape:
                raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
            view1 = arg1.ravel()
            view2 = arg2.ravel()
            outputObj = np.empty_like(arg1)
            output = outputObj.ravel()
            for i in range(view1.shape[0]):
                output[i] = func(self,view1[i],view2[i])
            return outputObj
        if tArg1 is np.ndarray and tArg2 is not np.ndarray:
            view1 = arg1.ravel()
            outputObj = np.empty_like(arg1)
            output = outputObj.ravel()
            dArg2 = arg2
            for i in range(view1.shape[0]):
                output[i] = func(self,view1[i],dArg2)
            return outputObj
        if tArg2 is np.ndarray and tArg1 is not np.ndarray:
            view2 = arg2.ravel()
            outputObj = np.empty_like(arg2)
            output = outputObj.ravel()
            dArg1 = arg1
            for i in range(view2.shape[0]):
                output[i] = func(self,dArg1,view2[i])
            return outputObj
        dArg1, dArg2 = arg1, arg2
        return func(self,dArg1,dArg2)
    cdef object wrapIII(self, int (*func)(RNG, int, int), object arg1, object arg2):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef int[:] view1
        cdef int[:] view2
        cdef int[:] output
        cdef int dArg1
        cdef int dArg2
        cdef type tArg1 = type(arg1), tArg2 = type(arg2)
        if tArg1 is np.ndarray and tArg2 is np.ndarray:
            if arg1.shape != arg2.shape:
                raise ValueError("Operand shapes do not match: {}, {}".format(arg1.shape,arg2.shape))
            view1 = arg1.ravel()
            view2 = arg2.ravel()
            outputObj = np.empty_like(arg1)
            output = outputObj.ravel()
            for i in range(view1.shape[0]):
                output[i] = func(self,view1[i],view2[i])
            return outputObj
        if tArg1 is np.ndarray and tArg2 is not np.ndarray:
            view1 = arg1.ravel()
            outputObj = np.empty_like(arg1)
            output = outputObj.ravel()
            dArg2 = arg2
            for i in range(view1.shape[0]):
                output[i] = func(self,view1[i],dArg2)
            return outputObj
        if tArg2 is np.ndarray and tArg1 is not np.ndarray:
            view2 = arg2.ravel()
            outputObj = np.empty_like(arg2)
            output = outputObj.ravel()
            dArg1 = arg1
            for i in range(view2.shape[0]):
                output[i] = func(self,dArg1,view2[i])
            return outputObj
        dArg1, dArg2 = arg1, arg2
        return func(self,dArg1,dArg2)
