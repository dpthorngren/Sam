import sys
import re

try:
    filename = sys.argv[1]
    targetClass = sys.argv[2]
except:
    print "Must specify filename and target class."
    sys.exit()
inFile = open(filename)
pyxOut = open("distributions.pyx","w")
pxdOut = open("distributions.pxd","w")

sigs = set()
declarations = ""

pxdOut.write("""cdef extern from \"sam.h\":
    cdef cppclass RNG:
        RNG()
        RNG(unsigned int)
""")
pyxOut.write("""
cdef class RandomNumberGenerator:
""")

for i, line in enumerate(inFile):
    if targetClass+"::" not in line:
        continue
    line = line.strip()
    line = re.sub(targetClass+"::",'',line)
    line = re.sub("min",'xMin',line)
    line = re.sub("max",'xMax',line)
    if line.startswith(targetClass):
        continue
    outputType = line[:line.find(' ')]
    functionName = line[line.find(' ')+1:line.find('(')]
    arguments = line[line.find('(')+1:line.find(')')].split(',')
    arguments = [a.strip() for a in arguments]
    argTypes = [a.split(' ')[0] for a in arguments]
    argNames = [a.split(' ')[1] for a in arguments]
    sig = (outputType[0]+"".join([c[0] for c in argTypes])).upper()
    sigs.add(sig)
    pxdOut.write("        "+outputType+" "+functionName+"("+", ".join(arguments)+") except +\n")
    declarations += "    cdef "+outputType+" _"+functionName+"(self, "+", ".join(arguments)+")\n"
    declarations += "    cpdef object "+functionName+"(self, object "+", object ".join(argNames)+")\n"
    pyxOut.write("    cdef "+outputType+" _"+functionName+"(self, "+", ".join(arguments)+"):\n")
    pyxOut.write("        return self."+targetClass.lower()+"."+functionName+"("+", ".join(argNames)+")\n")
    pyxOut.write("    cpdef object "+functionName+"(self, object "+", object ".join(argNames)+"):\n")
    pyxOut.write("        return self.wrap"+sig+"(RandomNumberGenerator._"+functionName+","+", ".join(argNames)+")\n")

pxdOut.write("""
cdef class RandomNumberGenerator:
    cdef RNG rng
""")
pxdOut.write(declarations)

# TYPECODE, RETTYPE, ARG1TYPE, ARG2TYPE, ARG3TYPE
wrapCode3 = """    cdef object wrapTYPECODE(self, RETTYPE (*func)(RandomNumberGenerator, ARG1TYPE, ARG2TYPE, ARG3TYPE), object arg1, object arg2, object arg3):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef ARG1TYPE[:] view1
        cdef ARG1TYPE cArg1
        cdef ARG2TYPE[:] view2
        cdef ARG2TYPE cArg2
        cdef ARG3TYPE[:] view3
        cdef ARG3TYPE cArg3
        cdef RETTYPE[:] output
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
                else: # array, array, ARG3TYPE
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
                if tArg3 is np.ndarray: # array, ARG2TYPE, array
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
                else: # array, ARG2TYPE, ARG3TYPE
                    view1 = arg1.ravel()
                    cArg2, cArg3 = arg2, arg3
                    outputObj = np.empty_like(arg1)
                    output = outputObj.ravel()
                    for i in range(view1.shape[0]):
                        output[i] = func(self,view1[i],cArg2,cArg3)
                    return outputObj
        else:
            if tArg2 is np.ndarray:
                if tArg3 is np.ndarray: # ARG1TYPE, array, array
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
                else: # ARG1TYPE, array, ARG3TYPE
                    cArg1, cArg3 = arg1, arg3
                    view2 = arg2.ravel()
                    outputObj = np.empty_like(arg2)
                    output = outputObj.ravel()
                    for i in range(view2.shape[0]):
                        output[i] = func(self,cArg1,view2[i],cArg3)
                    return outputObj
            else:
                if tArg3 is np.ndarray: # ARG1TYPE, ARG2TYPE, array
                    cArg1, cArg2 = arg1, arg2
                    view3 = arg3.ravel()
                    outputObj = np.empty_like(arg3)
                    output = outputObj.ravel()
                    for i in range(view3.shape[0]):
                        output[i] = func(self,cArg1,cArg2,view3[i])
                    return outputObj
                else: # ARG1TYPE, ARG2TYPE, ARG3TYPE
                    cArg1, cArg2, cArg3 = arg1, arg2, arg3
                    return func(self,cArg1,cArg2,cArg3)
"""

# TYPECODE, RETTYPE, ARG1TYPE, ARG2TYPE
wrapCode2 = """    cdef object wrapTYPECODE(self, RETTYPE (*func)(RandomNumberGenerator, ARG1TYPE, ARG2TYPE), object arg1, object arg2):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef ARG1TYPE[:] view1
        cdef ARG2TYPE[:] view2
        cdef RETTYPE[:] output
        cdef ARG1TYPE dArg1
        cdef ARG2TYPE dArg2
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
"""

# TYPECODE, RETTYPE, ARGTYPE
wrapCode1 = """    cdef object wrapTYPECODE(self, RETTYPE (*func)(RandomNumberGenerator, ARGTYPE), object arg1):
        cdef Py_ssize_t i
        cdef object outputObj
        cdef ARGTYPE[:] view
        cdef RETTYPE[:] output
        if type(arg1) is np.ndarray:
            view = arg1.ravel()
            outputObj = np.empty_like(arg1)
            output = outputObj.ravel()
            for i in range(view.shape[0]):
                output[i] = func(self,view[i])
            return outputObj
        return func(self,<ARGTYPE>arg1)
"""

a = list(sigs)
a.sort()
td = {'D':"double","I":"int"}
for i in a:
    ret = td[i[0]]
    argsNoName = [td[j] for j in i[1:]]
    argsName = ["arg"+str(n+1) for n, j in enumerate(i[1:])]
    args = [td[j]+" arg"+str(n+1) for n, j in enumerate(i[1:])]
    if len(args) == 3:
        code = re.sub("TYPECODE",i,wrapCode3)
        code = re.sub("RETTYPE",ret,code)
        code = re.sub("ARG1TYPE",argsNoName[0],code)
        code = re.sub("ARG2TYPE",argsNoName[1],code)
        code = re.sub("ARG3TYPE",argsNoName[2],code)
        pyxOut.write(code)
        pxdOut.write("    cdef object wrap"+i+"(self, "+ret+" (*func)(RandomNumberGenerator, "+", ".join(argsNoName)+"), object "+", object ".join(argsName)+")\n")
    if len(args) == 2:
        code = re.sub("TYPECODE",i,wrapCode2)
        code = re.sub("RETTYPE",ret,code)
        code = re.sub("ARG1TYPE",argsNoName[0],code)
        code = re.sub("ARG2TYPE",argsNoName[1],code)
        pyxOut.write(code)
        pxdOut.write("    cdef object wrap"+i+"(self, "+ret+" (*func)(RandomNumberGenerator, "+", ".join(argsNoName)+"), object "+", object ".join(argsName)+")\n")
    if len(args) == 1:
        code = re.sub("TYPECODE",i,wrapCode1)
        code = re.sub("RETTYPE",ret,code)
        code = re.sub("ARGTYPE",argsNoName[0],code)
        pyxOut.write(code)
        pxdOut.write("    cdef object wrap"+i+"(self, "+ret+" (*func)(RandomNumberGenerator, "+", ".join(argsNoName)+"), object "+argsName[0]+")\n")

pyxOut.close()
pxdOut.close()
