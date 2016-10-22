from distutils.core import setup, Extension

samcpp = Extension('sam', sources=['sam.cpp'])

setup(name='Sam',
      description='An MCMC sampling system for c++, with bindings to python.',
      ext_modules=[samcpp])
