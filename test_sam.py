import unittest
import sam
from math import log, sqrt
import numpy as np


class SamTester(unittest.TestCase):
    # ===== Special Functions =====
    def testSpecialFunctions(self):
        self.assertAlmostEqual(sam.incBeta(.8,3.4,2.1),.04811402)
        self.assertAlmostEqual(sam.beta(.7,2.5),0.7118737432)
        self.assertAlmostEqual(sam.gamma(2.5),1.329340388)
        self.assertAlmostEqual(sam.digamma(12.5),2.4851956512)
        return

    # ===== Distributions =====
    def testNormalDistribution(self):
        self.assertAlmostEqual(sam.normalPDF(1,3,4),0.08801633)
        self.assertAlmostEqual(sam.normalMean(2,4),2.)
        self.assertAlmostEqual(sam.normalVar(2,4),16.)
        self.assertAlmostEqual(sam.normalStd(2,4),4.)
        self.assertAlmostEqual(sam.normalLogPDF(1,3,4),log(0.08801633))
        a = [sam.normalRand(3,2) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),3.,delta=3*.01)

    def testUniformDistribution(self):
        self.assertAlmostEqual(sam.uniformMean(2,4),3.)
        self.assertAlmostEqual(sam.uniformVar(2,4),4./12.)
        self.assertAlmostEqual(sam.uniformStd(2,4),2./sqrt(12.))
        self.assertAlmostEqual(sam.uniformPDF(3,2,4),0.5)
        self.assertAlmostEqual(sam.uniformLogPDF(3,2,4),log(0.5))
        self.assertAlmostEqual(sam.uniformCDF(2.5,2,4),0.25)
        a = [sam.uniformRand(3,4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),3.5,delta=3.5*.01)

    def testGammaDistribution(self):
        self.assertAlmostEqual(sam.gammaMean(3,4),.75)
        self.assertAlmostEqual(sam.gammaVar(3,4),3./16)
        self.assertAlmostEqual(sam.gammaStd(3,4),sqrt(3)/4.)
        self.assertAlmostEqual(sam.gammaPDF(1,3,4),.586100444)
        self.assertAlmostEqual(sam.gammaLogPDF(1,3,4),log(.586100444))
        self.assertAlmostEqual(sam.gammaCDF(1,3,4),0.7618966944464)
        a = [sam.gammaRand(3,4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),3./4,delta=.75*.01)

    def testInvGammaDistribution(self):
        self.assertAlmostEqual(sam.invGammaMean(3,4),2.)
        self.assertAlmostEqual(sam.invGammaVar(3,4),4.)
        self.assertAlmostEqual(sam.invGammaStd(3,4),2.)
        self.assertAlmostEqual(sam.invGammaPDF(1,3,4),.0060843811)
        self.assertAlmostEqual(sam.invGammaLogPDF(1,3,4),log(.0060843811))
        self.assertAlmostEqual(sam.invGammaCDF(1,3,4),.002161,delta=.001)
        a = [sam.invGammaRand(3,4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),2.,delta=2*.01)

    def testBetaDistribution(self):
        self.assertAlmostEqual(sam.betaMean(3,4),3./7)
        self.assertAlmostEqual(sam.betaVar(3,4),.0306122)
        self.assertAlmostEqual(sam.betaStd(3,4),0.17496355305)
        self.assertAlmostEqual(sam.betaPDF(.5,3,4),1.875)
        self.assertAlmostEqual(sam.betaLogPDF(.5,3,4),log(1.875))
        self.assertAlmostEqual(sam.betaCDF(.5,3,4),.65625)
        a = [sam.betaRand(3,4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),3./7,delta=3./7.*.01)

    def testPoissonDistribution(self):
        self.assertAlmostEqual(sam.poissonMean(2.4),2.4)
        self.assertAlmostEqual(sam.poissonVar(2.4),2.4)
        self.assertAlmostEqual(sam.poissonStd(2.4),sqrt(2.4))
        self.assertAlmostEqual(sam.poissonPDF(3,2.4),.2090141643)
        self.assertAlmostEqual(sam.poissonLogPDF(3,2.4),log(.2090141643))
        self.assertAlmostEqual(sam.poissonCDF(3.2,2.4),0.7787229)
        a = [sam.poissonRand(3.4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),3.4,delta=3.4*.01)

    def testExponentialDistribution(self):
        self.assertAlmostEqual(sam.exponentialMean(2.4),1./2.4)
        self.assertAlmostEqual(sam.exponentialVar(2.4),2.4**-2)
        self.assertAlmostEqual(sam.exponentialStd(2.4),1./2.4)
        self.assertAlmostEqual(sam.exponentialPDF(1,2.4),0.2177230878)
        self.assertAlmostEqual(sam.exponentialLogPDF(1,2.4),log(0.2177230878))
        self.assertAlmostEqual(sam.exponentialCDF(1,2.4),0.9092820)
        a = [sam.exponentialRand(3.4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),1./3.4,delta=1./3.4 * .01)

    def testBinomialDistribution(self):
        self.assertAlmostEqual(sam.binomialMean(10,.4),4.)
        self.assertAlmostEqual(sam.binomialVar(10,.4),.4*.6*10.)
        self.assertAlmostEqual(sam.binomialStd(10,.4),sqrt(.4*.6*10.))
        self.assertAlmostEqual(sam.binomialPDF(3,10,.4),.2149908)
        self.assertAlmostEqual(sam.binomialLogPDF(3,10,.4),-1.53715981920)
        self.assertAlmostEqual(sam.binomialCDF(3.4,10,.4),0.3822806)
        a = [sam.binomialRand(10,.74) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a),7.4,delta=7.4*.01)

if __name__ == "__main__":
    unittest.main()
