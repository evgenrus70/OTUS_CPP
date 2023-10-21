#include "Layer.h"
#include <functional>
#include <vector>

Layer::Layer (std::string _name, int _numLayer, int _inFm, int _outFm, int _inW) {
    name = _name;
    numLayer = _numLayer;
    inW = _inW;
    inFm = _inFm;
    outFm = _outFm;
    //inputData = vector_3d(inW,inW,inFm);
    //outputData = vector_3d(inW,inW,outFm);

    vector_3d<int> _inputData (inW,inW,inFm);
    vector_3d<int> _outputData (inW,inW,outFm);
    inputData = _inputData;
    outputData = _outputData;

}

void Layer::print(){
    std::cout << name <<std::endl;
}

void Layer::printInputs(){
    /*for (int i = 0; i < inW; i++) {
        for (int j = 0; j < inW; j++) {
            for (int k = 0; k < inFm; k++) {
                std::cout << inputData(i,j,k);
            }
        }
    }*/
    std::cout << inputData(0,0,0);
    std::cout<<"\n";
}

void Layer::printOutputs(){
    /*for (int i = 0; i < inW; i++) {
        for (int j = 0; j < inW; j++) {
            for (int k = 0; k < outFm; k++) {
                std::cout << outputData(i,j,k);
            }
        }
    }*/
    std::cout << outputData(0,0,0);
    std::cout<<"\n";
}

void Layer::forward () {
    vector_4d<int> output(64,3,256,256);
    vector_4d<float> weights(64,3,3,3);
    //std::transform(vec4d.begin(), vec4d.end(), vec4d.begin(), incrementor<int>(0));
    //std::copy(vec4d.begin(), vec4d.end(), std::ostream_iterator<int>(std::cout, " "));
    //std::cout << "\n";
    
    /*for (int ix = 0; ix < 64; ix++) {
        for (int iy = 0; iy < 3; iy++) {
            for (int x = 0; x < 256; x++) {
                for (int y = 0; y < 256; y++) {
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            output(x,y,ix,iy) += inputData.at<Vec3b>(i+x, j+y)[iy] * weights(ix,iy,i,j);
                        }
                        
                    }
                }
            }
        }
    }*/
    for (int i = 0; i < inW; i++) {
        for (int j = 0; j < inW; j++) {
            for (int k = 0; k < inFm; k++) {
                outputData(i,j,k) = inputData(i,j,k) + numLayer;
            }
        }
    }
    
}
