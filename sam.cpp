#include "sam.h"

// ===== Base Sampler Methods =====
std::string BaseSampler::getStatus(){
    std::string name = this->name;
    if(name.empty()){
        name = std::string("AnonymousSampler");
    }
    return name + std::string(": No further information implemented.");
}

void BaseSampler::sample(){
    std::cout << "Warning: BaseSampler::sample called." << std::endl;
    return;
}

// ===== Sam Methods =====
Sam::Sam(size_t nDim, double (*logLike)(double*, size_t)){
    this->nDim = nDim;
    this->logLike = logLike;
    samples = NULL;
    // Declare x and working memory in a contiguous block
    x = new double[4*nDim];
    propose = &x[nDim*sizeof(double)];
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
        status << samplers[i]->getStatus();
    return std::string(status.str());
}
