import unittest
import pysam
from math import log, sqrt
import numpy as np


class SamTester(unittest.TestCase):
    def setUp(self):
        self.rng = pysam.RandomNumberGenerator()
        return

    # ===== Special Functions =====
    def testSpecialFunctions(self):
        self.assertAlmostEqual(pysam.incBeta(.8,3.4,2.1),.04811402)
        self.assertAlmostEqual(pysam.beta(.7,2.5),0.7118737432)
        self.assertAlmostEqual(pysam.gamma(2.5),1.329340388)
        self.assertAlmostEqual(pysam.digamma(12.5),2.4851956512)

    # ===== Distributions =====
    def testNormalDistribution(self):
        self.assertAlmostEqual(self.rng.normalPDF(1,3,4),0.08801633)
        self.assertAlmostEqual(self.rng.normalLogPDF(1,3,4),log(0.08801633))
        a = [self.rng.normalRand(3,2) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),3.,delta=3*.01)

    def testUniformDistribution(self):
        # self.assertAlmostEqual(self.rng.uniformMean(2,4),3.)
        # self.assertAlmostEqual(self.rng.uniformVar(2,4),4./12.)
        # self.assertAlmostEqual(self.rng.uniformStd(2,4),2./sqrt(12.))
        self.assertAlmostEqual(self.rng.uniformPDF(3,2,4),0.5)
        self.assertAlmostEqual(self.rng.uniformLogPDF(3,2,4),log(0.5))
        # self.assertAlmostEqual(self.rng.uniformCDF(2.5,2,4),0.25)
        a = [self.rng.uniformRand(3,4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),3.5,delta=3.5*.01)

    def testGammaDistribution(self):
        # self.assertAlmostEqual(self.rng.gammaMean(3,4),.75)
        # self.assertAlmostEqual(self.rng.gammaVar(3,4),3./16)
        # self.assertAlmostEqual(self.rng.gammaStd(3,4),sqrt(3)/4.)
        self.assertAlmostEqual(self.rng.gammaPDF(1,3,4),.586100444)
        self.assertAlmostEqual(self.rng.gammaLogPDF(1,3,4),log(.586100444))
        # self.assertAlmostEqual(self.rng.gammaCDF(1,3,4),.7618966)
        a = [self.rng.gammaRand(3,4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),3./4,delta=.75*.01)

    # InvGamma distribution
    def testInvGammaDistribution(self):
        # self.assertAlmostEqual(self.rng.invGammaMean(3,4),2.)
        # self.assertAlmostEqual(self.rng.invGammaVar(3,4),4.)
        # self.assertAlmostEqual(self.rng.invGammaStd(3,4),2.)
        self.assertAlmostEqual(self.rng.invGammaPDF(1,3,4),.0060843811)
        self.assertAlmostEqual(self.rng.invGammaLogPDF(1,3,4),log(.0060843811))
        # self.assertAlmostEqual(self.rng.invGammaCDF(1,3,4),.002161,.001)
        a = [self.rng.invGammaRand(3,4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),2.,delta=2*.01)

    # Beta distribution
    def testBetaDistribution(self):
        # self.assertAlmostEqual(self.rng.betaMean(3,4),3./7)
        # self.assertAlmostEqual(self.rng.betaVar(3,4),.0306122)
        # self.assertAlmostEqual(self.rng.betaStd(3,4),0.17496355305)
        self.assertAlmostEqual(self.rng.betaPDF(.5,3,4),1.875)
        self.assertAlmostEqual(self.rng.betaLogPDF(.5,3,4),log(1.875))
        # self.assertAlmostEqual(self.rng.betaCDF(.5,3,4),.65625)
        a = [self.rng.betaRand(3,4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),3./7,delta=3./7.*.01)

    # Poisson distribution
    def testPoissonDistribution(self):
        # self.assertAlmostEqual(self.rng.poissonMean(2.4),2.4)
        # self.assertAlmostEqual(self.rng.poissonVar(2.4),2.4)
        # self.assertAlmostEqual(self.rng.poissonStd(2.4),sqrt(2.4))
        self.assertAlmostEqual(self.rng.poissonPDF(3,2.4),.2090141643)
        self.assertAlmostEqual(self.rng.poissonLogPDF(3,2.4),log(.2090141643))
        # self.assertAlmostEqual(self.rng.poissonCDF(3.2,2.4),0.7787229)
        a = [self.rng.poissonRand(3.4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),3.4,delta=3.4*.01)

    # Exponential distribution
    def testExponentialDistribution(self):
        # self.assertAlmostEqual(self.rng.exponentialMean(2.4),1./2.4)
        # self.assertAlmostEqual(self.rng.exponentialVar(2.4),2.4**-2)
        # self.assertAlmostEqual(self.rng.exponentialStd(2.4),1./2.4)
        self.assertAlmostEqual(self.rng.exponentialPDF(1,2.4),0.2177230878)
        self.assertAlmostEqual(self.rng.exponentialLogPDF(1,2.4),log(0.2177230878))
        # self.assertAlmostEqual(self.rng.exponentialCDF(1,2.4),0.9092820)
        a = [self.rng.exponentialRand(3.4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),1./3.4,delta=1./3.4 * .01)

    # # Binomial distribution
    def testBinomialDistribution(self):
        # self.assertAlmostEqual(self.rng.binomialMean(10,.4),4.)
        # self.assertAlmostEqual(self.rng.binomialVar(10,.4),.4*.6*10.)
        # self.assertAlmostEqual(self.rng.binomialStd(10,.4),sqrt(.4*.6*10.))
        self.assertAlmostEqual(self.rng.binomialPDF(3,10,.4),.2149908)
        self.assertAlmostEqual(self.rng.binomialLogPDF(3,10,.4),-1.53715981920)
        # self.assertAlmostEqual(self.rng.binomialCDF(3.4,10,.4),0.3822806)
        a = [self.rng.binomialRand(10,.74) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),7.4,delta=7.4*.01)

if __name__ == "__main__":
    unittest.main()
