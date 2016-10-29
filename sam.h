#ifndef sam_h__
#define sam_h__
#include <string>
#include <sstream>
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
    double normalRand(double=0, double=1);
    double normalPDF(double,double=0,double=1);
    double normalLogPDF(double, double=0, double=1);
    // Uniform Distribution
    double uniformRand(double=0, double=1);
    double uniformPDF(double,double=0,double=1);
    double uniformLogPDF(double,double=0,double=1);
    // Integer Uniform Distribution
    int uniformIntRand(int=0, int=1);
    double uniformIntPDF(int,int=0,int=1);
    double uniformIntLogPDF(int,int=0,int=1);
    // Gamma Distribution
    double gammaRand(double, double);
    double gammaPDF(double, double, double);
    double gammaLogPDF(double, double, double);
    // Inverse Gamma Distribution
    double invGammaRand(double, double);
    double invGammaPDF(double, double, double);
    double invGammaLogPDF(double, double, double);
    // Beta Distribution
    double betaRand(double, double);
    double betaPDF(double, double, double);
    double betaLogPDF(double, double, double);
    // Poisson Distribution
    int poissonRand(double);
    double poissonPDF(int, double);
    double poissonLogPDF(int, double);
    // Exponential Distribution
    double exponentialRand(double);
    double exponentialPDF(double, double);
    double exponentialLogPDF(double, double);
    // Binomial Distribution
    int binomialRand(int, double);
    double binomialPDF(int, int, double);
    double binomialLogPDF(int, int, double);
};

// Interface.  Derived classes below.
class BaseSampler{
public:
    Sam *man;
    RNG rng;
    double (*logProb)(double*);
    size_t targetStart;
    size_t targetStop;
    virtual ~BaseSampler() = 0;
    virtual BaseSampler* copyToHeap()=0;
    virtual std::string getStatus() = 0;
    virtual void sample() = 0;
};

class Metropolis: public BaseSampler{
public:
    double* proposalStd;
    int numProposed;
    int numAccepted;
    Metropolis();
    Metropolis(Sam*, size_t, size_t, double*,
        double (*logProb)(double*)=NULL);
    ~Metropolis();
    BaseSampler* copyToHeap();
    std::string getStatus();
    void sample();
};

class Sam{
public:
    // Random Number Generator
    RNG rng;
    // Parameters
    double (*logProb)(double*);
    size_t nDim;
    // Working memory
    double* x;
    double* xPropose;
    double* working1;
    double* working2;
    //Output
    double* samples;
    // User-called Functions
    std::vector<BaseSampler*> samplers;
    Sam(size_t, double (*)(double*));
    Sam();
    ~Sam();
    void run(size_t,double*,size_t,size_t);
    std::string getStatus();
};

#endif // sam_h__
