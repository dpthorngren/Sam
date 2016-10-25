#include "sam.h"

// ===== Sam Methods =====
Sam::Sam(size_t nDim, double (*logProb)(double*)){
    this->nDim = nDim;
    this->logProb = logProb;
    samples = NULL;
    // Declare x and working memory in a contiguous block
    x = new double[4*nDim];
    xPropose = &x[nDim*sizeof(double)];
    working1 = &x[2*nDim*sizeof(double)];
    working2 = &x[3*nDim*sizeof(double)];
    return;
}

Sam::~Sam(){
    // Deallocate samplers
    for(size_t i=0; i < samplers.size(); i++){
        if(samplers[i] != NULL){
            delete samplers[i];
            samplers[i] = NULL;
        }
    }
    // Deallocate working memory
    delete[] x;
    return;
}

void Sam::run(size_t nSamples, double* x0, size_t burnIn, size_t thin){
    size_t i, j, k;
    // Allocate memory for recording data.
    if(samples != NULL){
        delete[] samples;
        samples = NULL;
    }
    samples = new double[nSamples*nDim];

    // Set initial position
    for(i = 0; i < nDim; i++)
        x[i] = x0[i];

    // Burn-in
    for(i = 0; i < burnIn; i++){
        for(k = 0; k < samplers.size(); k++){
            samplers[k]->sample();
        }
    }

    // Collect data
    for(i = 0; i < nSamples; i++){
        for(j = 0; j <= thin; j++){
            for(k = 0; k < samplers.size(); k++){
                samplers[k]->sample();
            }
        }
        // Record data.
        for(j = 0; j < nDim; j++){
            samples[i*nDim + j] = x[j];
        }
    }
    return;
}

std::string Sam::getStatus(){
    std::stringstream status;
    status << "===== Sam Information =====" << std::endl;
    status << "Dimensions: " << nDim << std::endl;
    status << "Number of samplers:" << samplers.size() << std::endl;
    if(nDim < 20){
        status << "Data: ";
        for(size_t i = 0; i < nDim; i++){
            status << x[i] << " ";
        }
        status << std::endl;
    }
    else
        status << "Data: [Too much to reasonably print.]" << std::endl;
    status << "===== Sampler Information =====" << std::endl;
    for(size_t i = 0; i < samplers.size(); i++)
        status << i << " - " << samplers[i]->getStatus();
    return std::string(status.str());
}

// ===== BaseSampler Methods =====
BaseSampler::~BaseSampler(){
    return;
}

// ===== Metropolis Methods =====
Metropolis::Metropolis(){
    man = NULL;
    proposalStd = NULL;
    targetStart = 0;
    targetStop = 0;
    return;
}

Metropolis::Metropolis(Sam *man, size_t targetStart, size_t targetStop,
        double *proposalStd, double (*logProb)(double*)){
    this->man = man;
    if(logProb == NULL)
        this->logProb = man->logProb;
    else
        this->logProb = logProb;
    this->targetStart = targetStart;
    this->targetStop = targetStop;
    this->proposalStd = new double[targetStop-targetStart];
    for(size_t i = 0; i < (targetStop-targetStart); i++){
        this->proposalStd[i] = proposalStd[i];
    }
    return;
}

BaseSampler* Metropolis::copyToHeap(){
    Metropolis* heapCopy = new Metropolis;
    heapCopy->man = man;
    heapCopy->logProb = logProb;
    heapCopy->targetStart = targetStart;
    heapCopy->targetStop = targetStop;
    // Proper deep copy.  This will be less efficient in practice,
    // but far more intuitive for the user, and helps avoid memory leaks.
    heapCopy->proposalStd = new double[targetStop-targetStart];
    for(size_t i = 0; i < (targetStop-targetStart); i++){
        heapCopy->proposalStd[i] = proposalStd[i];
    }
    return static_cast<BaseSampler*>(heapCopy);
}

Metropolis::~Metropolis(){
    if(proposalStd != NULL){
        delete[] proposalStd;
        proposalStd = NULL;
    }
    return;
}

std::string Metropolis::getStatus(){
    std::stringstream status;
    status << "Metropolis:" << std::endl;
    bool isReady = true;
    if(proposalStd == NULL) isReady = false;
    if(man == NULL){
        isReady = false;
    }
    else{
        if(man->x == NULL) isReady = false;
        if(man->xPropose == NULL) isReady = false;
    }
    if(isReady) status << "    Initialized: True" << std::endl;
    else status << "    Initialized: False" << std::endl;
    status << "    Target: (" << targetStart << ":" << targetStop
           << "), Length = " << targetStop - targetStart << std::endl;
    status << "    Acceptance: " << numAccepted << " / " << numProposed
           << " (" << (double)numAccepted/numProposed << ")" << std::endl;
    return std::string(status.str());
}

void Metropolis::sample(){
    size_t i;
    for(i = 0; i < man->nDim; i++){
        man->xPropose[i] = man->x[i];
        if(i>=targetStart and i < targetStop){
            man->xPropose[i] = man->rng.normalRand(man->x[i],proposalStd[i-targetStart]);
        }
    }
    if(log(man->rng.uniformRand()) < (logProb(man->xPropose)-logProb(man->x))){
        for(i = targetStart; i < targetStop; i++)
            man->x[i] = man->xPropose[i];
            numAccepted += 1;
    }
    numProposed += 1;
    return;
}

// ===== Random Number Generator Methods =====
RNG::RNG(){
    mTwister.seed(static_cast<unsigned int>(std::time(0)));
    return;
}

RNG::RNG(unsigned int seed){
    mTwister.seed(seed);
    return;
}

double RNG::normalRand(double mean, double std){
    return normal_gen(mTwister)*std + mean;
}

double RNG::uniformRand(double min, double max){
    return uniform_gen(mTwister)*(max-min) + min;
}

int RNG::uniformIntRand(int min, int max){
    uniform_int_gen.param(boost::random::uniform_int_distribution<int>::param_type(min,max));
    return uniform_int_gen(mTwister);
}

double RNG::gammaRand(double shape, double rate){
    gamma_gen.param(boost::random::gamma_distribution<double>::param_type(shape,1./rate));
    return gamma_gen(mTwister);
}

double RNG::betaRand(double alpha, double beta){
    beta_gen.param(boost::random::beta_distribution<double>::param_type(alpha,beta));
    return beta_gen(mTwister);
}

int RNG::poissonRand(double rate){
    poisson_gen.param(boost::random::poisson_distribution<int>::param_type(rate));
    return poisson_gen(mTwister);
}

double RNG::exponentialRand(double rate){
    exponential_gen.param(boost::random::exponential_distribution<double>::param_type(rate));
    return exponential_gen(mTwister);
}

int RNG::binomialRand(int number, double probability){
    binomial_gen.param(boost::random::binomial_distribution<int,double>::param_type(number,probability));
    return binomial_gen(mTwister);
}
