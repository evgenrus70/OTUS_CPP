#include "Layer.h"

Layer::Layer (std::string _name) {
    name = _name;
}
void Layer::print(){
    std::cout << name <<std::endl;
}      

