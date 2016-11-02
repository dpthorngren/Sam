#include "sam.h"

// ===== Sam Methods =====
Sam::Sam(size_t nDim, double (*logProb)(double*)){
    this->nDim = nDim;
    this->logProb = logProb;
    samples = NULL;
    // Declare x and working memory in a contiguous block
    x = new double[2*nDim];
    xPropose = &x[nDim];
    return;
}

Sam::Sam(){
    nDim = 0;
    logProb = NULL;
    samples = NULL;
    x = NULL;
    xPropose = NULL;
    return;
}

Sam::~Sam(){
    // Deallocate working memory
    if(x != NULL){
        delete[] x;
        x = NULL;
    }
    for(size_t i = 0; i < subSamplers.size(); i++){
        if(subSamplers[i].dData != NULL){
            delete[] subSamplers[i].dData;
            subSamplers[i].dData = NULL;
        }
        if(subSamplers[i].sData != NULL){
            delete[] subSamplers[i].sData;
            subSamplers[i].sData = NULL;
        }
    }
    return;
}

double* Sam::getSamples(){
    return samples;
}

void Sam::write(std::string fileName, bool header, std::string sep){
    std::ofstream outputFile;
    outputFile.open(fileName.data());
    if(header){
        outputFile << getStatus();
    }
    if(samples==NULL){
        outputFile << "[No samples taken.]" << std::endl;
    }
    for(size_t i = 0; i < nSamples; i++){
        for(size_t j = 0; j < nDim; j++){
            outputFile << samples[i*nDim + j];
            if(j != nDim -1)
                outputFile << sep;
        }
        outputFile << std::endl;
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
    this->nSamples = nSamples;

    // Set initial position
    for(i = 0; i < nDim; i++)
        x[i] = x0[i];

    // Burn-in
    for(i = 0; i < burnIn; i++){
        for(k = 0; k < subSamplers.size(); k++){
            subSample(subSamplers[k]);
        }
    }

    // Collect data
    for(i = 0; i < nSamples; i++){
        for(j = 0; j <= thin; j++){
            for(k = 0; k < subSamplers.size(); k++){
                subSample(subSamplers[k]);
            }
        }
        // Record data.
        for(j = 0; j < nDim; j++){
            samples[i*nDim + j] = x[j];
        }
    }
    return;
}

void Sam::subSample(SubSamplerData &sub){
    switch(sub.algorithm){
        case NONE:
            // TODO: Error.
            return;
        case METROPOLIS:
            metropolisSample(sub);
            return;
        case GIBBS:
            gibbsSample(sub);
            return;
        case HAMILTONIAN:
            hamiltonianSample(sub);
            return;
        case CUSTOM:
            customSample(sub);
            return;
    }
    // TODO: Error
    return;
}

std::string Sam::subStatus(SubSamplerData &sub){
    switch(sub.algorithm){
        case NONE:
            // TODO: Error.
            return std::string("Error: No algorithm specified.");
        case METROPOLIS:
            return metropolisStatus(sub);
        case GIBBS:
            return gibbsStatus(sub);
        case HAMILTONIAN:
            return hamiltonianStatus(sub);
        case CUSTOM:
            return customStatus(sub);
    }
    // TODO: Error
    return std::string("Error: Algorithm not recognized.");
}

void Sam::metropolisSample(SubSamplerData& sub){
    size_t targetStart = sub.sData[0];
    size_t targetStop = targetStart + sub.sData[1];
    for(size_t i = 0; i < nDim; i++){
        if(i >= targetStart and i < targetStop){
            xPropose[i] = rng.normalRand(x[i],sub.dData[i-targetStart]);
        }
        else{
            xPropose[i] = x[i];
        }
    }
    // TODO: Record acceptance rate.
    proposeMetropolis();
    return;
}

bool Sam::proposeMetropolis(){
    // TODO: Add recording system for log probability.
    // TODO: Add alternative probability calculation option.
    double logRatio = logProb(xPropose) - logProb(x);
    if(log(rng.uniformRand(0,1)) < logRatio){
        for(size_t i = 0; i < nDim; i++)
            x[i] = xPropose[i];
        return true;
    }
    return false;
}

void Sam::gibbsSample(SubSamplerData& sub){
    return; // TODO: Implement
}

void Sam::hamiltonianSample(SubSamplerData& sub){
    return; // TODO: Implement
}

void Sam::customSample(SubSamplerData& sub){
    return; // TODO: Implement
}

std::string Sam::metropolisStatus(SubSamplerData& sub){
    std::stringstream status;
    status << "Metropolis: " << sub.sData[1] << " (" << sub.sData[0] << ":";
    status << sub.sData[0] + sub.sData[1] << ")" << std::endl;
    status << "Proposal: ";
    for(size_t i = 0; i < sub.dDataLen; i++){
        status << sub.dData[i];
        if(i != sub.dDataLen-1)
            status << ", ";
    }
    status << std::endl;
    return status.str();
    // return std::string("Error: Not Implemented."); // TODO: Implement
}

std::string Sam::gibbsStatus(SubSamplerData& sub){
    return std::string("Error: Not Implemented."); // TODO: Implement
}

std::string Sam::hamiltonianStatus(SubSamplerData& sub){
    return std::string("Error: Not Implemented."); // TODO: Implement
}

std::string Sam::customStatus(SubSamplerData& sub){
    return std::string("Error: Not Implemented."); // TODO: Implement
}

void Sam::addMetropolis(double* proposalStd, size_t targetStart, size_t targetLen){
    SubSamplerData newSub;
    newSub.algorithm = METROPOLIS;
    newSub.dData = new double[targetLen];
    newSub.dDataLen = targetLen;
    for(size_t i = 0; i < targetLen; i++)
        newSub.dData[i] = proposalStd[i];
    newSub.sData = new size_t[2];
    newSub.sDataLen = 2;
    newSub.sData[0] = targetStart;
    newSub.sData[1] = targetLen;
    subSamplers.push_back(newSub);
    return;
}

std::string Sam::getStatus(){
    std::stringstream status;
    status << "===== Sam Information =====" << std::endl;
    status << "Dimensions: " << nDim << std::endl;
    status << "Number of Sub-Samplers:" << subSamplers.size() << std::endl;
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
    for(size_t i = 0; i < subSamplers.size(); i++)
        status << i << " - " << subStatus(subSamplers[i]);
    return std::string(status.str());
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
