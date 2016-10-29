from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Workaround for the strict prototypes bug in distutils by subdir and daramarak at
# http://stackoverflow.com/questions/8106258/cc1plus-warning-command-line-option-wstrict-prototypes-is-valid-for-ada-c-o
import os
from distutils.sysconfig import get_config_vars
(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

print " ========== Setting up libsam (C++ library) ========== "
samcpp = Extension('libsam', sources=['sam.cpp'])
setup(name='sam',
      description='An MCMC sampling system for c++.',
      ext_modules=[samcpp])

print " ========== Setting up pysam (Python library) ========== "
setup(
    name="pysam",
    description="An MCMC sampling system written in C++ and wrapped to python.",
    ext_modules=cythonize('pysam.pyx'),
    include_dirs=[np.get_include()]
)
