from distutils.core import setup, Extension
from distutils.sysconfig import get_config_vars
from Cython.Build import cythonize
import numpy as np
import os

# Workaround for the strict prototypes bug in distutils by subdir and daramarak at
# http://stackoverflow.com/questions/8106258/cc1plus-warning-command-line-option-wstrict-prototypes-is-valid-for-ada-c-o
(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

ext = Extension("sam", ["src/sam.pyx"],
                extra_compile_args=["-Wno-cpp","-Wno-unused","-O3"])

setup(
    name="sam",
    ext_modules=cythonize([ext]),
    include_dirs=[np.get_include()],
)
