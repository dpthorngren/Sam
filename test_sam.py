import unittest
import sam
from math import log, sqrt
import numpy as np


class SamTester(unittest.TestCase):
    def test1DMetropolis(self):
        def logProb(x):
            return sam.betaLogPDF(x[0],15,20)
        a = sam.Sam(logProb,1,np.array([.5]),
                    lowerBoundaries=np.array([0.]),
                    upperBoundaries=np.array([1.]))
        a.addMetropolis(0,1)
        samples = a.run(20000,np.array([.5]))
        self.assertTrue((samples >= 0).all())
        self.assertTrue((samples <= 1).all())
        self.assertAlmostEqual(samples.mean(),sam.betaMean(15,20),delta=.01)
        self.assertAlmostEqual(samples.std(),sam.betaStd(15,20),delta=.01)

    def test2DMetropolis(self):
        def logProb(x):
            return sam.gammaLogPDF(x[0],20,40) + sam.normalLogPDF(x[1],5,1)
        a = sam.Sam(logProb,2,np.array([.5,.5]),
                    lowerBoundaries=np.array([0.,-np.inf]))
        a.addMetropolis(0,2)
        samples = a.run(50000,np.array([.5,.5]),1000)
        self.assertTrue((samples[:,0] >= 0).all())
        self.assertAlmostEqual(samples[:,0].mean(),sam.gammaMean(20,40),delta=.01)
        self.assertAlmostEqual(samples[:,0].std(),sam.gammaStd(20,40),delta=.01)
        self.assertAlmostEqual(samples[:,1].mean(),5.,delta=.1)
        self.assertAlmostEqual(samples[:,1].std(),1.,delta=.1)

    def test2DHMC(self):
        def logProb(x):
            return sam.gammaLogPDF(x[0],20,40) + sam.normalLogPDF(x[1],5,1)
        def gradLogProb(x):
            return np.array([sam.gammaDLDX(x[0],20,40), sam.normalDLDX(x[1],5,1)])
        a = sam.Sam(logProb,2,np.array([.5,.5]),
                    lowerBoundaries=np.array([0.,-np.inf]),
                    gradLogProbability=gradLogProb)
        a.addHMC(10,.1,0,2)
        samples = a.run(50000,np.array([.5,.5]),10)
        self.assertTrue((samples[:,0] >= 0).all())
        self.assertAlmostEqual(samples[:,0].mean(),sam.gammaMean(20,40),delta=.01)
        self.assertAlmostEqual(samples[:,0].std(),sam.gammaStd(20,40),delta=.01)
        self.assertAlmostEqual(samples[:,1].mean(),5.,delta=.1)
        self.assertAlmostEqual(samples[:,1].std(),1.,delta=.1)

    def test2DGradientDescent(self):
        def gradLogProb(x):
            return np.array([sam.gammaDLDX(x[0],20,40), sam.normalDLDX(x[1],5,1)])
        a = sam.Sam(None,2,np.array([.5,.5]),
                    lowerBoundaries=np.array([0.,-np.inf]),
                    gradLogProbability=gradLogProb)
        posteriorMax = a.gradientDescent(np.array([.5,.5]),step=.05)
        self.assertAlmostEqual(posteriorMax[0],19./40.,delta=1e-4)
        self.assertAlmostEqual(posteriorMax[1],5.,delta=1e-4)


class DistributionTester(unittest.TestCase):
    # ===== Special Functions =====
    def testSpecialFunctions(self):
        self.assertAlmostEqual(sam.incBeta(.8,3.4,2.1),.04811402)
        self.assertAlmostEqual(sam.beta(.7,2.5),0.7118737432)
        self.assertAlmostEqual(sam.gamma(2.5),1.329340388)
        self.assertAlmostEqual(sam.digamma(12.5),2.4851956512)

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


class GriddyTester(unittest.TestCase):
    def setUp(self):
        self.testF = lambda x, y: np.cos(x) + 2*y
        self.testGradF = lambda x, y: np.array([-np.sin(x),2])
        self.x = (np.linspace(0,10,1000),
                  np.sin(np.linspace(0,np.pi/2,900)))
        self.y = self.testF(self.x[0][:,np.newaxis],self.x[1][np.newaxis,:])
        self.a = sam.Griddy(self.x,self.y)

    def testStrides(self):
        self.assertEqual(self.a.getNPoints()[0], 1000)
        self.assertEqual(self.a.getNPoints()[1], 900)
        self.assertEqual(self.a.getStrides()[0], 900)
        self.assertEqual(self.a.getStrides()[1], 1)

    def testIndexing(self):
        self.assertEqual(len(self.a.getValues()),900000)
        self.assertEqual(self.a.ind(np.array([0,0],dtype=int)),0)
        self.assertEqual(self.a.ind(np.array([10,4],dtype=int)),9004)

    def testPointIdentification(self):
        # Point 1 (off grid in dimension 0)
        self.assertFalse(self.a.locatePoints(np.array([5,np.pi/4],dtype=np.double)))
        self.assertEqual(self.a.getIndices()[0], 499)
        self.assertEqual(self.a.getIndices()[1], 517)
        self.assertAlmostEqual(self.a.getWeights()[0], .5)
        self.assertAlmostEqual(self.a.getWeights()[1], .0001017340)
        # Point 2 (off grid in dimension 1)
        self.assertFalse(self.a.locatePoints(np.array([1,np.pi/8],dtype=np.double)))
        self.assertEqual(self.a.getIndices()[0], 99)
        self.assertEqual(self.a.getIndices()[1], 230)
        self.assertAlmostEqual(self.a.getWeights()[0], .9)
        self.assertAlmostEqual(self.a.getWeights()[1], 0.9685815061)
        # Point 3
        self.assertTrue(self.a.locatePoints(np.array([10,0],dtype=np.double)))
        self.assertEqual(self.a.getIndices()[0], 998)
        self.assertEqual(self.a.getIndices()[1], 0)
        self.assertAlmostEqual(self.a.getWeights()[0], .9)
        self.assertAlmostEqual(self.a.getWeights()[1], 1e-10)
        # Point 4
        self.assertTrue(self.a.locatePoints(np.array([0,np.pi/2],dtype=np.double)))
        self.assertEqual(self.a.getIndices()[0], 0)
        self.assertEqual(self.a.getIndices()[1], 898)
        self.assertAlmostEqual(self.a.getWeights()[0], 1e-10)
        self.assertAlmostEqual(self.a.getWeights()[1], 1e-10)

    def testGridValues(self):
        self.assertAlmostEqual(
            self.a.getValues()[self.a.ind(np.array([50,33]))],
            self.testF(self.x[0][50],np.sin(self.x[1][33])),
            delta=1e-4)

    def testInterpolation(self):
        self.assertAlmostEqual(
            self.a.interp(np.array([5,np.pi/4],dtype=np.double)),
            self.testF(5,np.pi/4),
            delta=1e-4)
        self.assertAlmostEqual(
            self.a.interp(np.array([1,np.pi/8],dtype=np.double)),
            self.testF(1,np.pi/8),
            delta=1e-4)
        self.assertTrue(np.isnan(self.a.interp(np.array([-1,np.pi/8],dtype=np.double))))

    def testGradientInterpolation(self):
        c = np.zeros(2)
        b = np.array([2.3,np.pi/6.4],dtype=np.double)
        self.a.interp(b,gradient=c)
        self.assertAlmostEqual(c[0], self.testGradF(b[0],b[1])[0], delta=.01)
        self.assertAlmostEqual(c[1], self.testGradF(b[0],b[1])[1], delta=.01)
        b = np.array([5,np.pi/4],dtype=np.double)
        self.a.interp(b,gradient=c)
        self.assertAlmostEqual(c[0], self.testGradF(b[0],b[1])[0], delta=.01)
        self.assertAlmostEqual(c[1], self.testGradF(b[0],b[1])[1], delta=.01)

    def testVectorizedInterp(self):
        b = np.array([[5,np.pi/4],[7.34,np.pi/6]],dtype=np.double)
        c = np.zeros(2)
        self.a.interpN(b,c)
        self.assertAlmostEqual(c[0], self.testF(5,np.pi/4),delta=1e-5)
        self.assertAlmostEqual(c[1], self.testF(7.34,np.pi/6),delta=1e-5)

if __name__ == "__main__":
    unittest.main()
