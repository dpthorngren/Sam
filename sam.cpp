#include "sam.h"

std::string BaseSampler::getStatus(){
    std::string name = this->name;
    if(name.empty()){
        name = std::string("AnonymousSampler");
    }
    return name + std::string(": No further information implemented.");
}

Sam::Sam(size_t nDim, double (*logLike)(double*, size_t)){
    this->nDim = nDim;
    this->logLike = logLike;
    return;
}

Sam::~Sam(){
    for(size_t i=0; i<this->samplers.size(); i++){
        if(samplers[i] != NULL){
            delete samplers[i];
            samplers[i] = NULL;
        }
    }
    return;
}

std::string Sam::getStatus(){
    std::stringstream status;
    status << "===== Sam Information =====" << std::endl;
    status << "Dimensions: " << this->nDim << std::endl;
    status << "Number of samplers:" << this->samplers.size() << std::endl;
    status << "===== Sampler Information =====" << std::endl;
    for(size_t i=0; i<this->samplers.size(); i++){
        status << samplers[i]->getStatus();
    }
    return std::string(status.str());
}
