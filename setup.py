from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extension = Extension("hmc",["hmc.pyx"],
                      include_dirs=[np.get_include()],
                      libraries=["gsl","gslcblas","m"])

setup(
    ext_modules=cythonize(extension)
)
