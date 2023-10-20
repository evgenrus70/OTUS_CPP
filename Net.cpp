#include "Net.h"

Net::Net (std::string _name,std::string _weightsPath,std::string _inputPath,std::string _maskPath) {
    name = _name;
    weightsPath = _weightsPath;
    inputPath = _inputPath;
    maskPath = _maskPath;
}
void Net::print(){
    std::cout << name <<std::endl;
    std::cout << weightsPath <<std::endl;
    std::cout << inputPath <<std::endl;
    std::cout << maskPath <<std::endl;
}      

void Net::openImage (std::string path) {
    
}

void Net::addLayer (Layer layer) {
    layers.push_back(layer);
}

void Net::printLayers () {
    for (const auto& layer : layers) {
        std::cout << layer.name << std::endl;
    }
}
