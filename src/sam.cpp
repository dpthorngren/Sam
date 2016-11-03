#include "sam.h"

// ===== Sam Methods =====
Sam::Sam(size_t nDim, double (*logProb)(double*)){
    setRecordOptions(true,true,false);
    this->nDim = nDim;
    this->logProb = logProb;
    samples = NULL;
    nSamples = 0;
    // Declare x and working memory in a contiguous block
    x = new double[2*nDim];
    xPropose = &x[nDim];
    acc = NULL;
    return;
}

Sam::Sam(){
    setRecordOptions(false,false,false);
    nDim = 0;
    logProb = NULL;
    samples = NULL;
    nSamples = 0;
    x = NULL;
    xPropose = NULL;
    acc = NULL;
    return;
}

Sam::~Sam(){
    // Deallocate working memory
    if(x != NULL){
        delete[] x;
        x = NULL;
    }
    if(acc != NULL){
        delete[] acc;
        acc = NULL;
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
        outputFile.close();
        return;
    }
    for(size_t i = 0; i < nSamples; i++){
        for(size_t j = 0; j < nDim; j++){
            outputFile << samples[i*nDim + j];
            if(j != nDim -1)
                outputFile << sep;
        }
        outputFile << std::endl;
    }
    outputFile.close();
    return;
}

void Sam::run(size_t nSamples, double* x0, size_t burnIn, size_t thin){
    size_t i, j, k;
    // Allocate memory for recording data.
    if(samples != NULL){
        delete[] samples;
        samples = NULL;
    }
    if(recordSamples)
        samples = new double[nSamples*nDim];
    this->nSamples = nSamples;
    if(acc != NULL){
        delete[] acc;
        acc = NULL;
    }
    acc = new Accumulator[nDim];

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
        record(i);
    }
    return;
}

void Sam::record(size_t i){
    for(size_t j = 0; j < nDim; j++){
        if(recordSamples)
            samples[i*nDim + j] = x[j];
        if(accumulateStats)
            acc[j](x[j]);
        if(printSamples)
            std::cout << x[j] << " ";
    }
    if(printSamples)
        std::cout << std::endl;
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

void Sam::setRecordOptions(bool recordSamples, bool accumulateStats, bool printSamples){
    this->recordSamples = recordSamples;
    this->accumulateStats = accumulateStats;
    this->printSamples = printSamples;
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
    if(accumulateStats){
        status << std::endl << "===== Variable summary =====" << std::endl;
        // TODO: Format output better.
        status << "Mean, Std:" << std::endl;
        for(size_t i = 0; i < subSamplers.size(); i++){
            status << getMean(i) << " " << getStd(i);
        }
    }
    status << std::endl << std::endl << "===== Sampler Information =====" << std::endl;
    for(size_t i = 0; i < subSamplers.size(); i++)
        status << i << " - " << subStatus(subSamplers[i]);
    return std::string(status.str());
}

double Sam::getMean(size_t i){
    return boost::accumulators::mean(acc[i]);
}

double Sam::getVar(size_t i){
    return boost::accumulators::variance(acc[i]);
}

double Sam::getStd(size_t i){
    return sqrt(boost::accumulators::variance(acc[i]));
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

double RNG::uniformCDF(double x, double min, double max){
    if(x <= min)
        return 0.0;
    if(x >= max)
        return 1.0;
    return (x-min)/(max-min);
}

// === Uniform Int Distribution ===
double RNG::uniformIntMean(int min, int max){
    return (min+max)/2.0;
}

double RNG::uniformIntVar(int min, int max){
    return ((max-min+1)*(max-min+1)-1)/12.;
}

double RNG::uniformIntStd(int min, int max){
    return sqrt(uniformIntVar(min,max));
}

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

double RNG::uniformIntCDF(double x, int min, int max){
    if(x < min)
        return 0.;
    if(x > max)
        return 1.;
    return (1+int(x)-min)/double(1+max-min);
    return 0.;
}

// === Gamma Distribution ===
double RNG::gammaMean(double shape, double rate){
    return shape/rate;
}

double RNG::gammaVar(double shape, double rate){
    return shape/(rate*rate);
}

double RNG::gammaStd(double shape, double rate){
    return sqrt(shape/(rate*rate));
}

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

double RNG::gammaCDF(double x, double shape, double rate){
    return cdf(boost::math::gamma_distribution<double>(shape,1./rate),x);
}

// === Inverse Gamma Distribution ===
double RNG::invGammaMean(double shape, double rate){
    return rate/(shape-1.);
}

double RNG::invGammaVar(double shape, double rate){
    return rate*rate / ((shape-1)*(shape-1)*(shape-2));
}

double RNG::invGammaStd(double shape, double rate){
    return sqrt(invGammaVar(shape,rate));
}

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

double RNG::invGammaCDF(double x, double shape, double rate){
    return cdf(boost::math::gamma_distribution<double>(shape,rate),1./x);
}


// === Beta Distribution ===
double RNG::betaMean(double alpha, double beta){
    return alpha / (alpha + beta);
}

double RNG::betaVar(double alpha, double beta){
    return alpha * beta / ((alpha+beta)*(alpha+beta)*(alpha+beta+1));
}

double RNG::betaStd(double alpha, double beta){
    return sqrt(betaVar(alpha,beta));
}

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

double RNG::betaCDF(double x, double alpha, double beta){
    return cdf(boost::math::beta_distribution<double>(alpha,beta),x);
}

// === Poisson Distribution ===
double RNG::poissonMean(double rate){
    return rate;
}

double RNG::poissonVar(double rate){
    return rate;
}

double RNG::poissonStd(double rate){
    return sqrt(rate);
}

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

double RNG::poissonCDF(double x, double rate){
    return cdf(boost::math::poisson_distribution<double>(rate),int(x));
}

// === Exponential Distribution ===
double RNG::exponentialMean(double rate){
    return 1./rate;
}

double RNG::exponentialVar(double rate){
    return 1./(rate*rate);
}

double RNG::exponentialStd(double rate){
    return 1./rate;
}

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

double RNG::exponentialCDF(double x, double rate){
    if(x > 0)
        return 1.-exp(-rate*x);
    return 0.;
}

// === Binomial Distribution ===
double RNG::binomialMean(int number, double probability){
    return number*probability;
}

double RNG::binomialVar(int number, double probability){
    return number*probability*(1.-probability);
}

double RNG::binomialStd(int number, double probability){
    return sqrt(number*probability*(1.-probability));
}

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

double RNG::binomialCDF(double x, int number, double probability){
    return cdf(boost::math::binomial_distribution<double>(number,probability),int(x));
}
