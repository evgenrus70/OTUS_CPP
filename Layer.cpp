#include "Layer.h"
#include <functional>
#include <vector>

Layer::Layer (std::string _type, int _numLayer, int _inFm, int _outFm, int _inW, int _coreW) {
    type = _type;
    numLayer = _numLayer;
    name = type + "_" + std::to_string(numLayer);
    inW = _inW;
    inFm = _inFm;
    outFm = _outFm;
    coreW = _coreW;
    //inputData = vector_3d(inW,inW,inFm);
    //outputData = vector_3d(inW,inW,outFm);

    vector_3d<int> _inputData (inW,inW,inFm);
    vector_3d<int> _outputData (inW,inW,outFm);
    vector_4d<int> _weights (inFm,outFm,coreW,coreW);
    inputData = _inputData;
    outputData = _outputData;
    weights = _weights;
}

void Layer::print(){
    std::cout << name <<std::endl;
}

void Layer::printInputs(){
    std::cout <<"layer " << numLayer << " inputs" << std::endl;
    for (int i = 0; i < inW; i++) {
        for (int j = 0; j < inW; j++) {
            for (int k = 0; k < inFm; k++) {
                std::cout << inputData(i,j,k);
            }
        }
    }    
    std::cout<<"\n";
}

void Layer::printOutputs(){
    std::cout <<"layer " << numLayer << " outputs" << std::endl;
    for (int i = 0; i < inW; i++) {
        for (int j = 0; j < inW; j++) {
            for (int k = 0; k < outFm; k++) {
                std::cout << outputData(i,j,k);
            }
        }
    }
    std::cout<<"\n";
}

void Layer::conv () {
    std::cout << name << std::endl;
    vector_4d<int> out_tmp (inW,inW,outFm,inFm); 
    int i = 0, j = 0;
    for (int ix = 0; ix < outFm; ix++) {
        for (int iy = 0; iy < inFm; iy++) {
            for (int x = 0; x < inW - 2; x++) {
                for (int y = 0; y < inW - 2; y++) {
                    for (i = 0; i < coreW; i++) {
                        for (j = 0; j < coreW; j++) {
                            out_tmp(x,y,ix,iy) +=  inputData(i+x, j+y, iy) * weights(ix,iy,i,j);
                        }
                    }
                    outputData(x,y,ix) += out_tmp(x,y,ix,iy);
                }
            }
        }
    }
}

void Layer::pool () {
    std::cout << name << std::endl;
}

void Layer::upsample(){
    std::cout << name << std::endl;
}

void Layer::forward () {
    if (!type.compare("conv")) 
        conv();
    else if (!type.compare("pool")) 
        pool();
    else if (!type.compare("upsample")) 
        upsample();
}
