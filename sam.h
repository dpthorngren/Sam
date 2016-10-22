#ifndef sam_h__
#define sam_h__
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <cmath>

class BaseSampler{
public:
    std::string name;
    std::string getStatus();
    void sample();
};

class Sam{
public:
    // Parameters
    size_t nDim;
    double (*logLike)(double*, size_t);
    std::vector<BaseSampler*> samplers;

    // User-called Functions
    Sam(size_t, double (*)(double*,size_t));
    ~Sam();
    std::string getStatus();
};

#endif // sam_h__
