from distutils.core import setup
from distutils.extension import Extension
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

extension = Extension("sammy",["sammy.pyx"],
                      include_dirs=[np.get_include()],
                      language="c++",
                      extra_compile_args=["-Wno-cpp","-Wno-unused"])

setup(
    ext_modules=cythonize(extension)
)
