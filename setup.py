from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension("sam", ["src/sam.pyx"],
                extra_compile_args=["-Wno-cpp", "-Wno-unused", "-O3"])

setup(
    name="sam",
    version='0.6',
    description="Sam is an MCMC sampling library for python, written in Cython.",
    author="Daniel Thorngren",
    license='MIT',
    ext_modules=cythonize([ext], compiler_directives={'embedsignature': True, 'language_level': "3"}),
    include_dirs=[np.get_include()],
    install_requires=[
        'cython',
        'scipy',
        'numpy'],
    test_suite="test_sam.SamTester"
)
