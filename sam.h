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

class Sam;

// Random Number Distribution Handling
class RNG{
private:
    boost::mt19937 mTwister;
    boost::normal_distribution<double> normal;
    boost::uniform_01<double> uniform;
public:
    RNG();
    RNG(unsigned int);
    double normalRand(double=0, double=1);
    double uniformRand(double=0, double=1);
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
    ~Sam();
    void run(size_t,double*,size_t,size_t);
    std::string getStatus();
};

#endif // sam_h__
