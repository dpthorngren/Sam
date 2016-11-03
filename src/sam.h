#ifndef sam_h__
#define sam_h__
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <cmath>
#include <ctime>
#include <boost/math/special_functions.hpp>
#include <boost/math/distributions.hpp>
#include <boost/random.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

class Sam;

// Random Number Distribution Handling
class RNG{
private:
    boost::mt19937 mTwister;
    boost::normal_distribution<double> normal_gen;
    boost::uniform_01<double> uniform_gen;
    boost::gamma_distribution<double> gamma_gen;
    boost::random::beta_distribution<double> beta_gen;
    boost::random::poisson_distribution<int> poisson_gen;
    boost::random::exponential_distribution<double> exponential_gen;
    boost::random::binomial_distribution<int,double> binomial_gen;
    boost::random::uniform_int_distribution<int> uniform_int_gen;
public:
    RNG();
    RNG(unsigned int);
    // Normal Distribution
    double normalMean(double=0,double=1);
    double normalVar(double=0,double=1);
    double normalStd(double=0,double=1);
    double normalRand(double=0, double=1);
    double normalPDF(double,double=0,double=1);
    double normalLogPDF(double, double=0, double=1);
    // Uniform Distribution
    double uniformMean(double=0,double=1);
    double uniformVar(double=0,double=1);
    double uniformStd(double=0,double=1);
    double uniformRand(double=0, double=1);
    double uniformPDF(double,double=0,double=1);
    double uniformCDF(double, double=0, double=1);
    double uniformLogPDF(double,double=0,double=1);
    // Integer Uniform Distribution
    double uniformIntMean(int=0,int=1);
    double uniformIntVar(int=0,int=1);
    double uniformIntStd(int=0,int=1);
    int uniformIntRand(int=0, int=1);
    double uniformIntPDF(int,int=0,int=1);
    double uniformIntLogPDF(int,int=0,int=1);
    double uniformIntCDF(double,int=0,int=1);
    // Gamma Distribution
    double gammaMean(double,double);
    double gammaVar(double,double);
    double gammaStd(double,double);
    double gammaRand(double, double);
    double gammaPDF(double, double, double);
    double gammaLogPDF(double, double, double);
    double gammaCDF(double, double, double);
    // Inverse Gamma Distribution
    double invGammaMean(double,double);
    double invGammaVar(double,double);
    double invGammaStd(double,double);
    double invGammaRand(double, double);
    double invGammaPDF(double, double, double);
    double invGammaLogPDF(double, double, double);
    double invGammaCDF(double, double, double);
    // Beta Distribution
    double betaMean(double,double);
    double betaVar(double,double);
    double betaStd(double,double);
    double betaRand(double, double);
    double betaPDF(double, double, double);
    double betaLogPDF(double, double, double);
    double betaCDF(double, double, double);
    // Poisson Distribution
    double poissonMean(double);
    double poissonVar(double);
    double poissonStd(double);
    int poissonRand(double);
    double poissonPDF(int, double);
    double poissonLogPDF(int, double);
    double poissonCDF(double, double);
    // Exponential Distribution
    double exponentialMean(double);
    double exponentialVar(double);
    double exponentialStd(double);
    double exponentialRand(double);
    double exponentialPDF(double, double);
    double exponentialLogPDF(double, double);
    double exponentialCDF(double, double);
    // Binomial Distribution
    double binomialMean(int, double);
    double binomialVar(int, double);
    double binomialStd(int, double);
    int binomialRand(int, double);
    double binomialPDF(int, int, double);
    double binomialLogPDF(int, int, double);
    double binomialCDF(double, int, double);
};

typedef boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::mean, boost::accumulators::tag::variance> > Accumulator;

typedef enum{
    NONE,
    METROPOLIS,
    GIBBS,
    HAMILTONIAN,
    CUSTOM
} SamplingAlgorithm;

typedef struct{
    SamplingAlgorithm algorithm;
    double *dData;
    size_t *sData;
    size_t dDataLen, sDataLen;
} SubSamplerData;

class Sam{
private:
    // Working memory
    double* x;
    double* xPropose;
    double* samples;
    size_t nSamples;
    std::vector<SubSamplerData> subSamplers;
    Accumulator *acc;
    // Parameters
    double (*logProb)(double*);
    size_t nDim;
    bool recordSamples;
    bool printSamples;
    bool accumulateStats;;
    // Helper functions
    void subSample(SubSamplerData&);
    std::string subStatus(SubSamplerData&);
    bool proposeMetropolis();
    void record(size_t);
    // Sampling Algorithms
    void metropolisSample(SubSamplerData&);
    void gibbsSample(SubSamplerData&);
    void hamiltonianSample(SubSamplerData&);
    void customSample(SubSamplerData&);
    // Printing Algorithms
    std::string metropolisStatus(SubSamplerData&);
    std::string gibbsStatus(SubSamplerData&);
    std::string hamiltonianStatus(SubSamplerData&);
    std::string customStatus(SubSamplerData&);
public:
    // Random Number Generator
    RNG rng;
    // User-called Functions
    Sam(size_t, double (*)(double*));
    Sam();
    ~Sam();
    void setRecordOptions(bool, bool, bool);
    void run(size_t,double*,size_t,size_t);
    std::string getStatus();
    double* getSamples();
    void write(std::string, bool, std::string);
    void addMetropolis(double*, size_t, size_t);
    double getMean(size_t);
    double getVar(size_t);
    double getStd(size_t);
};

#endif // sam_h__
