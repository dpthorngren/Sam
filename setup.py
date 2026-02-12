import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ext = Extension(
    "sam", ["src/sam/sam.pyx"],
                extra_compile_args=["-Wno-cpp", "-Wno-unused", "-O3"], include_dirs=[np.get_include()],)

directives = {
    'embedsignature': True,
    'language_level': "3",
}

setup(ext_modules=cythonize([ext], compiler_directives=directives))
