=======================
Distributions Functions
=======================

To help with constructing functions computing log-probabilities and gradients
thereof, Sam provides a large number of fast Cython functions.  For computing
gradients, the functions of the form ``distributionDLD_`` return the derivative
of the log-probability with respect to the given variable; e.g., ``normalDLDV``
gives the derivative of the normal distribution with respect to the variance
at the given position.

.. module:: sam

Uniform Distribution
====================
.. autofunction:: uniformRand
.. autofunction:: uniformPDF
.. autofunction:: uniformLogPDF
.. autofunction:: uniformCDF
.. autofunction:: uniformDLDU
.. autofunction:: uniformDLDL
.. autofunction:: uniformMean
.. autofunction:: uniformVar
.. autofunction:: uniformStd

Normal Distribution
===================
.. autofunction:: normalRand
.. autofunction:: normalPDF
.. autofunction:: normalLogPDF
.. autofunction:: normalCDF
.. autofunction:: normalDLDM
.. autofunction:: normalDLDX
.. autofunction:: normalDLDV
.. autofunction:: normalDLDS
.. autofunction:: normalMean
.. autofunction:: normalVar
.. autofunction:: normalStd
.. autofunction:: normalMode

Multivariate Normal Distribution
================================
.. autofunction:: mvNormalRand
.. autofunction:: mvNormalPDF
.. autofunction:: mvNormalLogPDF

Gamma Distribution
==================
.. autofunction:: gammaRand
.. autofunction:: gammaPDF
.. autofunction:: gammaLogPDF
.. autofunction:: gammaCDF
.. autofunction:: gammaDLDA
.. autofunction:: gammaDLDB
.. autofunction:: gammaDLDX
.. autofunction:: gammaMean
.. autofunction:: gammaVar
.. autofunction:: gammaStd
.. autofunction:: gammaMode

Inverse Gamma Distribution
==========================
.. autofunction:: invGammaRand
.. autofunction:: invGammaPDF
.. autofunction:: invGammaLogPDF
.. autofunction:: invGammaCDF
.. autofunction:: invGammaDLDA
.. autofunction:: invGammaDLDB
.. autofunction:: invGammaDLDX
.. autofunction:: invGammaMean
.. autofunction:: invGammaVar
.. autofunction:: invGammaStd
.. autofunction:: invGammaMode

Beta Distribution
=================
.. autofunction:: betaRand
.. autofunction:: betaPDF
.. autofunction:: betaLogPDF
.. autofunction:: betaCDF
.. autofunction:: betaDLDA
.. autofunction:: betaDLDB
.. autofunction:: betaMean
.. autofunction:: betaVar
.. autofunction:: betaStd
.. autofunction:: betaMode

Poisson Distribution
====================
.. autofunction:: poissonRand
.. autofunction:: poissonPDF
.. autofunction:: poissonLogPDF
.. autofunction:: poissonCDF
.. autofunction:: poissonDLDL
.. autofunction:: poissonMean
.. autofunction:: poissonVar
.. autofunction:: poissonStd
.. autofunction:: poissonMode

Exponential Distribution
========================
.. autofunction:: exponentialRand
.. autofunction:: exponentialPDF
.. autofunction:: exponentialLogPDF
.. autofunction:: exponentialCDF
.. autofunction:: exponentialDLDL
.. autofunction:: exponentialMean
.. autofunction:: exponentialVar
.. autofunction:: exponentialStd
.. autofunction:: exponentialMode

Binomial Distribution
=====================
.. autofunction:: binomialRand
.. autofunction:: binomialPDF
.. autofunction:: binomialLogPDF
.. autofunction:: binomialCDF
.. autofunction:: binomialDLDP
.. autofunction:: binomialMean
.. autofunction:: binomialVar
.. autofunction:: binomialStd
.. autofunction:: binomialMode
