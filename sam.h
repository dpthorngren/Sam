#ifndef sam_h__
#define sam_h__
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <cmath>

class BaseSampler{
public:
    double* target;
    size_t targetLen;
    std::string name;
    std::string getStatus();
    void sample();
};

class Sam{
private:
    // Parameters
    double (*logLike)(double*, size_t);
    size_t nDim;
    // Working memory
    double* x;
    double* propose;
    double* working1;
    double* working2;
public:
    //Output
    double* samples;
    // User-called Functions
    std::vector<BaseSampler*> samplers;
    Sam(size_t, double (*)(double*,size_t));
    ~Sam();
    void run(size_t,double*,size_t,size_t);
    std::string getStatus();
};

#endif // sam_h__
