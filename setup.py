from distutils.core import setup
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

setup(
    name="sam",
    ext_modules=cythonize('sam.pyx'),
    include_dirs=[np.get_include()]
)
