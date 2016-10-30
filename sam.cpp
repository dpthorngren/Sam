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

Sam::Sam(){
    nDim = 0;
    logProb = NULL;
    samples = NULL;
    // Declare x and working memory in a contiguous block
    x = NULL;
    xPropose = NULL;
    working1 = NULL;
    working2 = NULL;
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
    if(x != NULL){
        delete[] x;
        x = NULL;
    }
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

// === Normal Distribution ===
double RNG::normalMean(double mean, double std){
    return mean;
}

double RNG::normalVar(double mean, double std){;
    return std*std;
}

double RNG::normalStd(double mean, double std){
    return std;
}

double RNG::normalRand(double mean, double std){
    return normal_gen(mTwister)*std + mean;
}

double RNG::normalPDF(double x, double mean, double std){
    return exp(-pow((x-mean)/std,2)/2.)/sqrt(2.*M_PI*std*std);
}

double RNG::normalLogPDF(double x, double mean, double std){
    return -pow((x-mean)/std,2)/2. - .5*log(2.*M_PI*std*std);
}

// === Uniform Distribution ===
double RNG::uniformMean(double min, double max){
    return (min+max)/2.0;
}

double RNG::uniformVar(double min, double max){
    return pow(max-min,2)/12.;
}

double RNG::uniformStd(double min, double max){
    return max-min/sqrt(12);
}

double RNG::uniformRand(double min, double max){
    return uniform_gen(mTwister)*(max-min) + min;
}

double RNG::uniformPDF(double x, double min, double max){
    if(x >= min and x <= max){
        return 1./(max-min);
    }
    return 0.;
}

double RNG::uniformLogPDF(double x, double min, double max){
    if(x >= min and x <= max){
        return -log(max-min);
    }
    return -INFINITY;
}

// === Uniform Int Distribution ===
int RNG::uniformIntRand(int min, int max){
    uniform_int_gen.param(boost::random::uniform_int_distribution<int>::param_type(min,max));
    return uniform_int_gen(mTwister);
}

double RNG::uniformIntPDF(int x, int min, int max){
    if(x >= min and x <= max){
        return 1./(1+max-min);
    }
    return 0.;
}

double RNG::uniformIntLogPDF(int x, int min, int max){
    if(x >= min and x <= max){
        return -log(1+max-min);
    }
    return INFINITY;
}

// === Gamma Distribution ===
double RNG::gammaRand(double shape, double rate){
    gamma_gen.param(boost::random::gamma_distribution<double>::param_type(shape,1./rate));
    return gamma_gen(mTwister);
}

double RNG::gammaPDF(double x, double shape, double rate){
    return pdf(boost::math::gamma_distribution<double>(shape,1./rate),x);
}

double RNG::gammaLogPDF(double x, double shape, double rate){
    //TODO: Hand-optimize.
    return log(this->gammaPDF(x,shape,rate));
}

// === Inverse Gamma Distribution ===
double RNG::invGammaRand(double shape, double rate){
    gamma_gen.param(boost::random::gamma_distribution<double>::param_type(shape,1./rate));
    return 1./gamma_gen(mTwister);
}

double RNG::invGammaPDF(double x, double shape, double rate){
    return pdf(boost::math::gamma_distribution<double>(shape,rate),1./x);
}

double RNG::invGammaLogPDF(double x, double shape, double rate){
    //TODO: Hand-optimize.
    return log(this->invGammaPDF(x,shape,rate));
}


// === Beta Distribution ===
double RNG::betaRand(double alpha, double beta){
    beta_gen.param(boost::random::beta_distribution<double>::param_type(alpha,beta));
    return beta_gen(mTwister);
}

double RNG::betaPDF(double x, double alpha, double beta){
    return pdf(boost::math::beta_distribution<double>(alpha,beta),x);
}

double RNG::betaLogPDF(double x, double alpha, double beta){
    //TODO: Hand-optimize.
    return log(this->betaPDF(x,alpha,beta));
}

// === Poisson Distribution ===
int RNG::poissonRand(double rate){
    poisson_gen.param(boost::random::poisson_distribution<int>::param_type(rate));
    return poisson_gen(mTwister);
}

double RNG::poissonPDF(int x, double rate){
    return pdf(boost::math::poisson_distribution<double>(rate),x);
}

double RNG::poissonLogPDF(int x, double rate){
    return log(this->poissonPDF(x,rate));
}

// === Exponential Distribution ===
double RNG::exponentialRand(double rate){
    exponential_gen.param(boost::random::exponential_distribution<double>::param_type(rate));
    return exponential_gen(mTwister);
}

double RNG::exponentialPDF(double x, double rate){
    if(x > 0)
        return rate * exp(-rate*x);
    return 0.;
}

double RNG::exponentialLogPDF(double x, double rate){
    if(x>0)
        return log(rate) - rate*x;
    return -INFINITY;
}

// === Binomial Distribution ===
int RNG::binomialRand(int number, double probability){
    binomial_gen.param(boost::random::binomial_distribution<int,double>::param_type(number,probability));
    return binomial_gen(mTwister);
}

double RNG::binomialPDF(int x, int number, double probability){
    return pdf(boost::math::binomial_distribution<double>(number,probability),x);
}

double RNG::binomialLogPDF(int x, int number, double probability){
    return log(this->binomialPDF(x,number,probability));
}
