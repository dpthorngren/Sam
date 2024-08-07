import unittest
import sam
from math import log, sqrt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logit


def logProb1(x, gradient, getGradient):
    if getGradient:
        gradient[0] = sam.gammaDLDX(x[0], 20, 40)
        gradient[1] = sam.normalDLDX(x[1], 5, 1)
    return sam.gammaLogPDF(x[0], 20, 40) + sam.normalLogPDF(x[1], 5, 1)


def logProb2(x):
    return sam.betaLogPDF(x[0], 15, 20)


def logProb3(x, gradient, getGradient):
    assert not getGradient
    return sam.betaLogPDF(x[0], 20, 40) + sam.normalLogPDF(x[1], 5, 1)


_logProb4_ = multivariate_normal(cov=[[1., .3], [.3, 1]]).logpdf


def logProb4(x):
    return _logProb4_(x)


def logProb5(x, params):
    return sam.normalLogPDF(x[0], params[0], params[1])


def logProb6(x, params, gradient, getGradient):
    if getGradient:
        gradient[0] = sam.betaDLDX(x[0], params[0], params[1])
    return sam.betaLogPDF(x[0], params[0], params[1])


def raisesLogProb(x):
    if x > np.inf:
        raise ValueError("x can never be good enough!")
    return -1


class SamTester(unittest.TestCase):
    def testErrorHandling(self):
        a = sam.Sam(raisesLogProb, [.5, .5], [0., -np.inf])
        self.assertIsNone(a.results)
        self.assertIsNone(a.samples)
        self.assertRaises(AssertionError, a.getStats)
        self.assertRaises(AssertionError, a.summary)
        self.assertRaises(ValueError, a.run, 1000, [.5, .5])
        self.assertRaises(AttributeError, a.gradientDescent, [.5, .5])
        self.assertRaises(ValueError, a.simulatedAnnealing, [.5, .5])
        self.assertRaises(AssertionError, a.getSampler, 2)
        self.assertRaises(OverflowError, a.getSampler, -3)
        self.assertRaises(ValueError, sam.normalCDF, 1, 0, -1)

    def testModelSelection(self):
        # This is a roundabout way to test them, but it does work

        def rightModel(x):
            return sam.normalLogPDF(x[0], 0, 1.)

        def wrongModel(x):
            return sam.normalLogPDF(x[0], 0, 2.)

        def flatPrior(x):
            return 0.

        a = sam.Sam(rightModel, .5)
        a.run(100000, .5, showProgress=False)
        b = sam.Sam(wrongModel, .5)
        b.run(100000, .5, showProgress=False)
        assert not any(np.isnan(a.resultsLogProb))
        assert not any(np.isnan(b.resultsLogProb))

        # DIC
        right = a.getDIC(flatPrior)
        wrong = b.getDIC(flatPrior)
        self.assertLessEqual(right, wrong)
        self.assertAlmostEqual(right, 3., delta=.2)
        self.assertAlmostEqual(wrong, 4.4, delta=.2)
        # AIC
        right = a.getAIC(flatPrior)
        wrong = b.getAIC(flatPrior)
        self.assertLessEqual(right, wrong)
        self.assertAlmostEqual(right, 3.837, delta=.01)
        self.assertAlmostEqual(wrong, 5.224, delta=.01)
        # BIC
        right = a.getBIC(flatPrior, 1000)
        wrong = b.getBIC(flatPrior, 1000)
        self.assertLessEqual(right, wrong)
        self.assertAlmostEqual(right, 8.74, delta=.01)
        self.assertAlmostEqual(wrong, 10.13, delta=.01)
        return

    def testACF(self):
        x = [np.pi]
        for i in range(10000):
            x.append(np.pi + .9*x[-1] + sam.normalRand())
        sampleACF = sam.acf(x, 30)
        theoryACF = .9**np.arange(30)
        self.assertTrue(np.allclose(sampleACF, theoryACF, .1, .1))
        return

    def testLogit(self):
        x = [.234124, 1.-1e-13, 1e-13]
        self.assertAlmostEqual(sam.logit(x[0]), logit(x[0]), 13)
        self.assertAlmostEqual(sam.logit(x[1]), logit(x[1]), 13)
        self.assertAlmostEqual(sam.logit(x[2]), logit(x[2]), 13)
        return

    def testGaussianProcess(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        y2 = np.cos(x)
        f = sam.GaussianProcess(x, y, None, 'exp')
        loglike = f.logLikelihood(np.array([10, .5, 0]))
        gpMean, gpVar = f.predict(np.array([5.]))
        gpVar = np.sqrt(np.diag(gpVar))
        with self.assertRaises(ValueError):
            f.gradient(3.5)
        self.assertAlmostEqual(gpMean[0], -0.957698488, delta=.01)
        self.assertAlmostEqual(gpVar[0], 0.0502516, delta=.01)
        self.assertAlmostEqual(loglike, 109.90324, delta=.01)
        f.setY(y2)
        gpMean = f.predict(np.array([5.]), False)
        self.assertAlmostEqual(gpMean[0], np.cos(5.), delta=.01)

    def testGaussianProcess2D(self):
        x = np.linspace(0, 1, 400).reshape(200, 2)
        z = np.sin(np.sum(x, axis=-1))
        f = sam.GaussianProcess(x, z, None, 'matern32')
        loglike = f.logLikelihood(np.array([1, .5, 0]))
        gpMean, gpVar = f.predict([[.5, .5]])
        gpVar = np.sqrt(np.diag(gpVar))
        grad = f.gradient([.5, .5])
        self.assertAlmostEqual(grad[0], 0.537, delta=.01)
        self.assertAlmostEqual(grad[1], 0.542, delta=.01)
        self.assertAlmostEqual(loglike, 1107.363, delta=.01)
        self.assertAlmostEqual(gpMean[0], 0.841, delta=.01)
        self.assertAlmostEqual(gpVar[0], 0.00217, delta=.01)

    def test1DMetropolis(self):
        a = sam.Sam(logProb2, .5, 0., 1.)
        samples = a.run(100000, 1, showProgress=False)
        self.assertGreaterEqual(a.getAcceptance()[0], 0.)
        self.assertLessEqual(a.getAcceptance()[0], 1.)
        self.assertTrue((samples >= 0).all())
        self.assertTrue((samples <= 1).all())
        self.assertAlmostEqual(samples.mean(), sam.betaMean(15, 20), delta=.01)
        self.assertAlmostEqual(samples.std(), sam.betaStd(15, 20), delta=.01)

    def testSummary(self):
        a = sam.Sam(logProb2, .5, 0., 1.)
        with self.assertRaises(AssertionError):
            a.summary()
        a.run(100000, .5, showProgress=False)
        self.assertGreaterEqual(len(a.summary(None, True)), 0)

    def testGetCovar(self):
        a = sam.Sam(logProb4, np.ones(2))
        a.addMetropolis()
        c = a.getProposalCov()
        for i, j in zip(c.flatten(), [1, 0., 0., 1]):
            self.assertAlmostEqual(i, j)
        a.clearSamplers()
        a.addMetropolis(np.array([[1, .1], [.1, 1.]])/2.)
        c = a.getProposalCov(0)
        for i, j in zip(c.flatten(), np.array([1, .1, .1, 1])/2.):
            self.assertAlmostEqual(i, j)
        a.clearSamplers()
        a.addHMC(10, .1)
        c = a.getProposalCov()
        for i, j in zip(c.flatten(), [1, 0., 0., 1]):
            self.assertAlmostEqual(i, j)
        a.clearSamplers()
        a.addAdaptiveMetropolis(np.array([[1, .1], [.1, 1.]])/2.)
        c = a.getProposalCov(0)
        # The covariance output is the sample covariance, which should be 0
        for i, j in zip(c.flatten(), [0, 0, 0, 0.]):
            self.assertAlmostEqual(i, j)

    def test2DMetropolis(self):
        a = sam.Sam(logProb1, [.5, .5], [0., -np.inf])
        samples = a.run(100000, [.5, .5], 1000, showProgress=False)
        self.assertGreaterEqual(a.getAcceptance()[0], 0.)
        self.assertLessEqual(a.getAcceptance()[0], 1.)
        self.assertGreaterEqual(a.getAcceptance()[1], 0.)
        self.assertLessEqual(a.getAcceptance()[1], 1.)
        self.assertTrue((samples[:, 0] >= 0).all())
        self.assertAlmostEqual(samples[:, 0].mean(), sam.gammaMean(20, 40), delta=.01)
        self.assertAlmostEqual(samples[:, 0].std(), sam.gammaStd(20, 40), delta=.01)
        self.assertAlmostEqual(samples[:, 1].mean(), 5., delta=.1)
        self.assertAlmostEqual(samples[:, 1].std(), 1., delta=.1)
        for i in range(50000):
            self.assertAlmostEqual(a.samplesLogProb[i], logProb1(a.samples[i], None, False))

    def testThreading(self):
        a = sam.Sam(logProb1, [.5, .5], lowerBounds=[0., -np.inf])
        samples = a.run(100000, [.5, .5], 1000, threads=5, showProgress=False)
        for i in a.getAcceptance():
            self.assertGreaterEqual(i[0], 0.)
            self.assertLessEqual(i[0], 1.)
            self.assertGreaterEqual(i[1], 0.)
            self.assertLessEqual(i[1], 1.)
        self.assertEqual(len(a.results.shape), 2)
        self.assertEqual(a.results.shape[0], 5*100000)
        self.assertEqual(a.results.shape[1], 2)
        self.assertEqual(len(a.samples.shape), 3)
        self.assertEqual(a.samples.shape[0], 5)
        self.assertEqual(a.samples.shape[1], 100000)
        self.assertEqual(a.samples.shape[2], 2)
        self.assertNotEqual(samples[0, -1, -1], samples[1, -1, -1])
        samples = np.concatenate([samples[0], samples[1]], axis=1)
        self.assertTrue((samples[:, 0] >= 0).all())
        self.assertAlmostEqual(samples[:, 0].mean(), sam.gammaMean(20, 40), delta=.01)
        self.assertAlmostEqual(samples[:, 0].std(), sam.gammaStd(20, 40), delta=.01)
        self.assertAlmostEqual(samples[:, 1].mean(), 5., delta=.1)
        self.assertAlmostEqual(samples[:, 1].std(), 1., delta=.1)
        for i in range(100000):
            for j in range(5):
                self.assertAlmostEqual(a.samplesLogProb[j, i],
                                       logProb1(a.samples[j, i], None, False))

    def testThreading2(self):
        a = sam.Sam(logProb1, [.5, .5], lowerBounds=[0., -np.inf])
        samples = a.run(100000, np.random.rand(5, 2), 1000, threads=5, showProgress=False)
        for i in a.getAcceptance():
            self.assertGreaterEqual(i[0], 0.)
            self.assertLessEqual(i[0], 1.)
            self.assertGreaterEqual(i[1], 0.)
            self.assertLessEqual(i[1], 1.)
        with self.assertRaises(AttributeError):
            a.samples = np.ones(5)
        self.assertEqual(samples.shape[0], 5)
        self.assertEqual(samples.shape[1], 100000)
        self.assertEqual(samples.shape[2], 2)
        self.assertNotEqual(samples[0, -1, -1], samples[1, -1, -1])
        samples = np.concatenate([samples[0], samples[1]], axis=1)
        self.assertTrue((samples[:, 0] >= 0).all())
        self.assertAlmostEqual(samples[:, 0].mean(), sam.gammaMean(20, 40), delta=.01)
        self.assertAlmostEqual(samples[:, 0].std(), sam.gammaStd(20, 40), delta=.01)
        self.assertAlmostEqual(samples[:, 1].mean(), 5., delta=.1)
        self.assertAlmostEqual(samples[:, 1].std(), 1., delta=.1)
        for i in range(len(a.resultsLogProb)):
            self.assertAlmostEqual(a.resultsLogProb[i], logProb1(a.results[i], None, False))

    def test2DHMC(self):
        a = sam.Sam(logProb1, [1, 1], lowerBounds=[0., -np.inf])
        a.addHMC(10, .1)
        samples = a.run(50000, [.5, .5], 10, showProgress=False)
        self.assertTrue((samples[:, 0] >= 0).all())
        self.assertAlmostEqual(samples[:, 0].mean(), sam.gammaMean(20, 40), delta=.05)
        self.assertAlmostEqual(samples[:, 0].std(), sam.gammaStd(20, 40), delta=.05)
        self.assertAlmostEqual(samples[:, 1].mean(), 5., delta=.2)
        self.assertAlmostEqual(samples[:, 1].std(), 1., delta=.2)

    def testCorrelatedMetropolis(self):
        a = sam.Sam(logProb4, np.ones(2))
        a.addMetropolis(np.array([[1, .1], [.1, 1.]])/2.)
        samples = a.run(100000, 5*np.ones(2), 1000, showProgress=False)
        self.assertAlmostEqual(samples[:, 0].mean(), 0., delta=.05)
        self.assertAlmostEqual(samples[:, 0].std(), 1., delta=.1)
        self.assertAlmostEqual(samples[:, 1].mean(), 0., delta=.05)
        self.assertAlmostEqual(samples[:, 1].std(), 1., delta=.1)

    def testAdaptiveMetropolis(self):
        a = sam.Sam(logProb4, np.ones(2))
        a.addAdaptiveMetropolis(np.array([[1, .1], [.1, 1.]])/2., scaling=4.)
        samples = a.run(50000, 5*np.ones(2), 1000, showProgress=False)
        self.assertAlmostEqual(samples[:, 0].mean(), 0., delta=.1)
        self.assertAlmostEqual(samples[:, 0].std(), 1., delta=.1)
        self.assertAlmostEqual(samples[:, 1].mean(), 0., delta=.1)
        self.assertAlmostEqual(samples[:, 1].std(), 1., delta=.1)

    def test2DGradientDescent(self):
        a = sam.Sam(logProb1, [.5, .5], lowerBounds=[0., -np.inf])
        posteriorMax = a.gradientDescent([.5, .5], step=.05)
        self.assertAlmostEqual(posteriorMax[0], 19./40., delta=1e-4)
        self.assertAlmostEqual(posteriorMax[1], 5., delta=1e-4)

    def testRunningStats(self):
        a = sam.Sam(logProb3, [.5, .5], lowerBounds=[0., -np.inf], upperBounds=[1., np.inf])
        a.addMetropolis()
        samples = a.run(100000, [.5, .5], 1000, recordStop=0, collectStats=True, showProgress=False)
        self.assertEqual(samples.size, 0)
        self.assertAlmostEqual(a.getStats()[0][0], sam.betaMean(20, 40), delta=.01)
        self.assertAlmostEqual(a.getStats()[1][0], sam.betaStd(20, 40), delta=.01)
        self.assertAlmostEqual(a.getStats()[0][1], 5, delta=.1)
        self.assertAlmostEqual(a.getStats()[1][1], 1, delta=.1)

    def testUserParams(self):
        a = sam.Sam(logProb5, [.4])
        a.addMetropolis()
        a.userParams = [0.5, 0.1]
        samples = a.run(10000, [0.5], showProgress=False)
        self.assertEqual(a.userParams[0], 0.5)
        self.assertEqual(a.userParams[1], 0.1)
        self.assertEqual(len(a.userParams), 2)
        self.assertAlmostEqual(samples[:, 0].mean(), 0.5, delta=.1)
        self.assertAlmostEqual(samples[:, 0].std(), 0.1, delta=.1)
        a.userParams = [15.1, 0.35]
        samples = a.run(10000, [14.], showProgress=False)
        self.assertEqual(a.userParams[0], 15.1)
        self.assertEqual(a.userParams[1], 0.35)
        self.assertAlmostEqual(samples[:, 0].mean(), 15.1, delta=.3)
        self.assertAlmostEqual(samples[:, 0].std(), 0.35, delta=.1)

    def testUserParamsThreaded(self):
        a = sam.Sam(logProb5, [.4])
        a.addMetropolis()
        a.userParams = [10.8, 0.25]
        a.run(10000, [14.], threads=4, showProgress=False)
        self.assertEqual(a.userParams[0], 10.8)
        self.assertEqual(a.userParams[1], 0.25)
        self.assertAlmostEqual(a.results[:, 0].mean(), 10.8, delta=.2)
        self.assertAlmostEqual(a.results[:, 0].std(), 0.25, delta=.1)

    def testUserParamsThreadedGradient(self):
        a = sam.Sam(logProb6, [.1], lowerBounds=[0.], upperBounds=[1.])
        a.addMetropolis()
        with self.assertRaises(AttributeError):
            a.run(10000, [.5], showProgress=False)
        a.userParams = [10., 15.]
        a.run(10000, [.5], threads=4, showProgress=False)
        self.assertEqual(a.userParams[0], 10.)
        self.assertEqual(a.userParams[1], 15.)
        self.assertAlmostEqual(a.results[:, 0].mean(), sam.betaMean(10., 15.), delta=.1)
        self.assertAlmostEqual(a.results[:, 0].std(), sam.betaStd(10., 15.), delta=.1)
        a.userParams = [65., 12.]
        a.run(10000, [.5], threads=4, showProgress=False)
        self.assertEqual(a.userParams[0], 65.)
        self.assertEqual(a.userParams[1], 12.)
        self.assertAlmostEqual(a.results[:, 0].mean(), sam.betaMean(65., 12.), delta=.1)
        self.assertAlmostEqual(a.results[:, 0].std(), sam.betaStd(65., 12.), delta=.1)

    def testScaleChange(self):
        # Intentionally bad scale to catch if it's not changed
        a = sam.Sam(logProb2, [1e-12], 0. ,1.)
        self.assertEqual(a.scale.ndim, 1)
        self.assertEqual(len(a.scale), 1)
        self.assertEqual(a.scale[0], 1e-12)
        a.scale = [0.5]
        samps0 = a.run(2500, [0.3], threads=4, showProgress=False)
        a.scale = [0.35]
        samps1 = a.run(2500, [0.5], threads=4, showProgress=False)
        self.assertAlmostEqual(samps0.mean(), samps1.mean(), delta=.01)
        self.assertAlmostEqual(samps0.std(), samps1.std(), delta=.1)
        with self.assertRaises(AssertionError):
            a.scale = [-15.]
        with self.assertRaises(AssertionError):
            a.scale = [.3, .2]
        with self.assertRaises(AssertionError):
            a.scale = [np.inf]

    def testBoundsChange(self):
        # Intentionally invalid bounds to catch if they aren't changed
        a = sam.Sam(logProb2, .5, -3.2, 1e5)
        self.assertEqual(a.lowerBounds[0], -3.2)
        self.assertEqual(a.upperBounds[0], 1e5)
        a.lowerBounds = [0.]
        a.upperBounds = [1.]
        samples = a.run(100000, 0.5, showProgress=False)
        self.assertGreaterEqual(a.getAcceptance()[0], 0.)
        self.assertLessEqual(a.getAcceptance()[0], 1.)
        self.assertTrue((samples >= 0).all())
        self.assertTrue((samples <= 1).all())
        self.assertAlmostEqual(samples.mean(), sam.betaMean(15, 20), delta=.01)
        self.assertAlmostEqual(samples.std(), sam.betaStd(15, 20), delta=.01)
        a.scale = [0.1]
        a.lowerBounds = [0.41]
        a.upperBounds = [0.49]
        samples = a.run(100000, 0.45, showProgress=False)
        self.assertTrue((samples >= .41).all())
        self.assertTrue((samples <= .49).all())

    def testExceptionsRaised(self):
        a = sam.Sam(None, np.ones(1))
        with self.assertRaises(RuntimeError):
            a(np.ones(1))


class DistributionTester(unittest.TestCase):
    # ===== Special Functions =====
    def testSpecialFunctions(self):
        self.assertAlmostEqual(sam.incBeta(.8, 3.4, 2.1), .04811402)
        self.assertAlmostEqual(sam.beta(.7, 2.5), 0.7118737432)
        self.assertAlmostEqual(sam.gamma(2.5), 1.329340388)
        self.assertAlmostEqual(sam.digamma(12.5), 2.4851956512)

    # ===== Distributions =====
    def testNormalDistribution(self):
        with self.assertRaises(ValueError):
            sam.normalPDF(0, 1, -3)
        with self.assertRaises(ValueError):
            sam.normalCDF(0., 1., 0.)
        with self.assertRaises(ValueError):
            sam.normalLogPDF(0, 1, -5.)
        self.assertAlmostEqual(sam.normalPDF(1, 3, 4), 0.08801633)
        self.assertAlmostEqual(sam.normalMean(2, 4), 2.)
        self.assertAlmostEqual(sam.normalVar(2, 4), 16.)
        self.assertAlmostEqual(sam.normalStd(2, 4), 4.)
        self.assertAlmostEqual(sam.normalLogPDF(1, 3, 4), log(0.08801633))
        a = [sam.normalRand(3, 2) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a), 3., delta=3*.01)

    def testMvNormalDistribution(self):
        targetCov = np.random.rand(3, 3)
        targetCov = targetCov*targetCov.T/2. + np.eye(3)
        a = np.empty((100000, 3))
        a = np.array([sam.mvNormalRand(np.array([1., 5., -3.]), targetCov) for i in range(100000)])
        self.assertAlmostEqual(np.mean(a[:, 0]), 1., delta=.05)
        self.assertAlmostEqual(np.mean(a[:, 1]), 5., delta=.05)
        self.assertAlmostEqual(np.mean(a[:, 2]), -3., delta=.05)
        for i, c in enumerate(np.cov(a.T, ddof=0).flatten()):
            self.assertAlmostEqual(targetCov.flatten()[i], c, delta=.05)
        targetChol = np.linalg.cholesky(targetCov)
        a = np.array([sam.mvNormalRand(np.array([1., 5., -3.]), targetChol, isChol=True)
                      for i in range(100000)])
        self.assertAlmostEqual(np.mean(a[:, 0]), 1., delta=.05)
        self.assertAlmostEqual(np.mean(a[:, 1]), 5., delta=.05)
        self.assertAlmostEqual(np.mean(a[:, 2]), -3., delta=.05)
        for i, c in enumerate(np.cov(a.T, ddof=0).flatten()):
            self.assertAlmostEqual(targetCov.flatten()[i], c, delta=.2)
        self.assertAlmostEqual(sam.mvNormalLogPDF(np.ones(3), np.zeros(3), targetCov.copy()),
                               multivariate_normal.logpdf(np.ones(3), np.zeros(3), targetCov))
        self.assertAlmostEqual(sam.mvNormalPDF(np.ones(3), np.zeros(3), targetCov.copy()),
                               multivariate_normal.pdf(np.ones(3), np.zeros(3), targetCov))

    def testUniformDistribution(self):
        self.assertAlmostEqual(sam.uniformMean(2, 4), 3.)
        self.assertAlmostEqual(sam.uniformVar(2, 4), 4./12.)
        self.assertAlmostEqual(sam.uniformStd(2, 4), 2./sqrt(12.))
        self.assertAlmostEqual(sam.uniformPDF(3, 2, 4), 0.5)
        self.assertAlmostEqual(sam.uniformLogPDF(3, 2, 4), log(0.5))
        self.assertAlmostEqual(sam.uniformCDF(2.5, 2, 4), 0.25)
        a = [sam.uniformRand(3, 4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a), 3.5, delta=3.5*.01)

    def testGammaDistribution(self):
        with self.assertRaises(ValueError):
            sam.gammaPDF(4., 1, -3)
        with self.assertRaises(ValueError):
            sam.gammaCDF(2., 0., 1.)
        with self.assertRaises(ValueError):
            sam.gammaMode(10., -np.inf)
        self.assertAlmostEqual(sam.gammaMean(3, 4), .75)
        self.assertAlmostEqual(sam.gammaVar(3, 4), 3./16)
        self.assertAlmostEqual(sam.gammaStd(3, 4), sqrt(3)/4.)
        self.assertAlmostEqual(sam.gammaPDF(1, 3, 4), .586100444)
        self.assertAlmostEqual(sam.gammaLogPDF(1, 3, 4), log(.586100444))
        self.assertAlmostEqual(sam.gammaCDF(1, 3, 4), 0.7618966944464)
        a = [sam.gammaRand(3, 4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a), 3./4, delta=.75*.01)

    def testInvGammaDistribution(self):
        with self.assertRaises(ValueError):
            sam.invGammaPDF(4., 1, -3)
        with self.assertRaises(ValueError):
            sam.invGammaCDF(2., 0., 1.)
        with self.assertRaises(ValueError):
            sam.invGammaMode(10., -np.inf)
        self.assertAlmostEqual(sam.invGammaMean(3, 4), 2.)
        self.assertAlmostEqual(sam.invGammaVar(3, 4), 4.)
        self.assertAlmostEqual(sam.invGammaStd(3, 4), 2.)
        self.assertAlmostEqual(sam.invGammaPDF(1, 3, 4), .0060843811)
        self.assertAlmostEqual(sam.invGammaLogPDF(1, 3, 4), log(.0060843811))
        self.assertAlmostEqual(sam.invGammaCDF(1, 3, 4), .002161, delta=.001)
        a = [sam.invGammaRand(3, 4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a), 2., delta=2*.01)

    def testBetaDistribution(self):
        with self.assertRaises(ValueError):
            sam.betaPDF(.3, 1, -3)
        with self.assertRaises(ValueError):
            sam.betaCDF(2., 0., 1.)
        with self.assertRaises(ValueError):
            sam.betaMode(10., -np.inf)
        self.assertAlmostEqual(sam.betaMean(3, 4), 3./7)
        self.assertAlmostEqual(sam.betaVar(3, 4), .0306122)
        self.assertAlmostEqual(sam.betaStd(3, 4), 0.17496355305)
        self.assertAlmostEqual(sam.betaPDF(.5, 3, 4), 1.875)
        self.assertAlmostEqual(sam.betaLogPDF(.5, 3, 4), log(1.875))
        self.assertAlmostEqual(sam.betaCDF(.5, 3, 4), .65625)
        a = [sam.betaRand(3, 4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a), 3./7, delta=3./7.*.01)

    def testPoissonDistribution(self):
        with self.assertRaises(ValueError):
            sam.poissonPDF(3, -1.5)
        with self.assertRaises(ValueError):
            sam.poissonStd(0.)
        with self.assertRaises(ValueError):
            sam.betaMode(-1., 3.)
        self.assertAlmostEqual(sam.poissonMean(2.4), 2.4)
        self.assertAlmostEqual(sam.poissonVar(2.4), 2.4)
        self.assertAlmostEqual(sam.poissonStd(2.4), sqrt(2.4))
        self.assertAlmostEqual(sam.poissonPDF(3, 2.4), .2090141643)
        self.assertAlmostEqual(sam.poissonLogPDF(3, 2.4), log(.2090141643))
        self.assertAlmostEqual(sam.poissonCDF(3.2, 2.4), 0.7787229)
        a = [sam.poissonRand(3.4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a), 3.4, delta=3.4*.01)

    def testExponentialDistribution(self):
        with self.assertRaises(ValueError):
            sam.exponentialPDF(3, -1.5)
        with self.assertRaises(ValueError):
            sam.exponentialStd(-3.)
        with self.assertRaises(ValueError):
            sam.exponentialMode(0.)
        self.assertAlmostEqual(sam.exponentialMean(2.4), 1./2.4)
        self.assertAlmostEqual(sam.exponentialVar(2.4), 2.4**-2)
        self.assertAlmostEqual(sam.exponentialStd(2.4), 1./2.4)
        self.assertAlmostEqual(sam.exponentialPDF(1, 2.4), 0.2177230878)
        self.assertAlmostEqual(sam.exponentialLogPDF(1, 2.4), log(0.2177230878))
        self.assertAlmostEqual(sam.exponentialCDF(1, 2.4), 0.9092820)
        a = [sam.exponentialRand(3.4) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a), 1./3.4, delta=1./3.4 * .01)

    def testBinomialDistribution(self):
        with self.assertRaises(ValueError):
            sam.binomialPDF(-3, -4, .6)
        with self.assertRaises(ValueError):
            sam.binomialVar(5, 1.1)
        with self.assertRaises(ValueError):
            sam.binomialMode(23, -.2)
        self.assertAlmostEqual(sam.binomialMean(10, .4), 4.)
        self.assertAlmostEqual(sam.binomialVar(10, .4), .4*.6*10.)
        self.assertAlmostEqual(sam.binomialStd(10, .4), sqrt(.4*.6*10.))
        self.assertAlmostEqual(sam.binomialPDF(3, 10, .4), .2149908)
        self.assertAlmostEqual(sam.binomialLogPDF(3, 10, .4), -1.53715981920)
        self.assertAlmostEqual(sam.binomialCDF(3.4, 10, .4), 0.3822806)
        a = [sam.binomialRand(10, .74) for i in range(100000)]
        self.assertAlmostEqual(np.mean(a), 7.4, delta=7.4*.01)


class GriddyTester(unittest.TestCase):
    def setUp(self):
        self.testF = lambda x, y: np.cos(x) + 2*y
        self.testGradF = lambda x, y: np.array([-np.sin(x), 2])
        self.x = (np.linspace(0, 10, 1000),
                  np.sin(np.linspace(0, np.pi/2, 900)))
        self.y = self.testF(self.x[0][:, None], self.x[1][None, :])
        self.a = sam.Griddy(self.x, self.y)

    def testStrides(self):
        self.assertEqual(self.a.getNPoints()[0], 1000)
        self.assertEqual(self.a.getNPoints()[1], 900)
        self.assertEqual(self.a.getStrides()[0], 900)
        self.assertEqual(self.a.getStrides()[1], 1)

    def testIndexing(self):
        self.assertEqual(len(self.a.getValues()), 900000)
        self.assertEqual(self.a.ind(np.array([0, 0], dtype=int)), 0)
        self.assertEqual(self.a.ind(np.array([10, 4], dtype=int)), 9004)

    def testPointIdentification(self):
        # Point 1 (off grid in dimension 0)
        self.assertFalse(self.a.locatePoints(np.array([5, np.pi/4], dtype=np.double)))
        self.assertEqual(self.a.getIndices()[0], 499)
        self.assertEqual(self.a.getIndices()[1], 517)
        self.assertAlmostEqual(self.a.getWeights()[0], .5)
        self.assertAlmostEqual(self.a.getWeights()[1], .0001017340)
        # Point 2 (off grid in dimension 1)
        self.assertFalse(self.a.locatePoints(np.array([1, np.pi/8], dtype=np.double)))
        self.assertEqual(self.a.getIndices()[0], 99)
        self.assertEqual(self.a.getIndices()[1], 230)
        self.assertAlmostEqual(self.a.getWeights()[0], .9)
        self.assertAlmostEqual(self.a.getWeights()[1], 0.9685815061)
        # Point 3
        self.assertTrue(self.a.locatePoints(np.array([10, 0], dtype=np.double)))
        self.assertEqual(self.a.getIndices()[0], 998)
        self.assertEqual(self.a.getIndices()[1], 0)
        self.assertAlmostEqual(self.a.getWeights()[0], .9)
        self.assertAlmostEqual(self.a.getWeights()[1], 1e-10)
        # Point 4
        self.assertTrue(self.a.locatePoints(np.array([0, np.pi/2], dtype=np.double)))
        self.assertEqual(self.a.getIndices()[0], 0)
        self.assertEqual(self.a.getIndices()[1], 898)
        self.assertAlmostEqual(self.a.getWeights()[0], 1e-10)
        self.assertAlmostEqual(self.a.getWeights()[1], 1e-10)

    def testGridValues(self):
        self.assertAlmostEqual(
            self.a.getValues()[self.a.ind(np.array([50, 33]))],
            self.testF(self.x[0][50], np.sin(self.x[1][33])),
            delta=1e-4)

    def testInterpolation(self):
        self.assertAlmostEqual(
            self.a.interp(np.array([5, np.pi/4], dtype=np.double)),
            self.testF(5, np.pi/4),
            delta=1e-4)
        self.assertAlmostEqual(
            self.a.interp(np.array([1, np.pi/8], dtype=np.double)),
            self.testF(1, np.pi/8),
            delta=1e-4)
        self.assertTrue(np.isnan(self.a.interp(np.array([-1, np.pi/8], dtype=np.double))))

    def testGradientInterpolation(self):
        c = np.zeros(2)
        b = np.array([2.3, np.pi/6.4], dtype=np.double)
        self.a.interp(b, gradient=c)
        self.assertAlmostEqual(c[0], self.testGradF(b[0], b[1])[0], delta=.01)
        self.assertAlmostEqual(c[1], self.testGradF(b[0], b[1])[1], delta=.01)
        b = np.array([5, np.pi/4], dtype=np.double)
        self.a.interp(b, gradient=c)
        self.assertAlmostEqual(c[0], self.testGradF(b[0], b[1])[0], delta=.01)
        self.assertAlmostEqual(c[1], self.testGradF(b[0], b[1])[1], delta=.01)

    def testVectorizedInterp(self):
        b = np.array([[5, np.pi/4], [7.34, np.pi/6]], dtype=np.double)
        c = np.zeros(2)
        self.a.interpN(b, c)
        self.assertAlmostEqual(c[0], self.testF(5, np.pi/4), delta=1e-5)
        self.assertAlmostEqual(c[1], self.testF(7.34, np.pi/6), delta=1e-5)


if __name__ == "__main__":
    unittest.main()
