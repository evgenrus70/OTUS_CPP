#include "Layer.h"
#include <functional>
#include <vector>
#include <chrono>
#include <stdlib.h>
#include <omp.h>

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
    std::cout<<"Start " << name << std::endl;
    const auto start{std::chrono::steady_clock::now()};
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
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    std::cout<< "End " << name <<" with time: " <<  elapsed_seconds.count() << std::endl;
}

void Layer::conv_gemm () {
    std::cout<<"Start gemm " << name << std::endl;
    const auto start{std::chrono::steady_clock::now()};
    int M = 64;
    int N1 = 258;
    int N2 = 258;
    int K = 27;
    int ALPHA = 1;
    int lda = 27;
    int ldb = 65536;
    int ldc1 = 65536;
    int M_start = 0;
    int pad_t = 1;
    int pad_l = 1;
    int ldc2 = 256;

    float *A = new float[64 * 3 * 3 * 3];       // weights
    float *B = new float[N1 * N1 * 3 * 3 * 3];  // image
    float *C = new float[N1 * N1 * 64];         // result

    int i,j,j1,j2,k;
    //#pragma omp parallel for num_threads(2)
    for(i = M_start; i < M_start+M; ++i){   // i = 0; i < 64; i++
        for(k = 0; k < K; ++k){             // k = 0; k < 27; ++k
            j = 0;                          
            float A_PART = A[i*lda+k]; // A_PART = A[0*27+0] : A[63*27+26];
            for(j1 = pad_t; j1 < N1; ++j1) {    // j1 = 1; j1 < 258; ++j1
                for(j2 = pad_l; j2 < N2; ++j2) {// j2 = 1; j2 < 258; ++j2
                    C[i*ldc1 + j1*ldc2 + j2] += A_PART*B[k*ldb+j]; // C[0*65536 + 1*256 + 1] += A_PART*B[0*65536 + 0] : C[63*65536 + 257*256 + 257]
                    j++;
                }
            }
        }
    }
    
    delete A;
    delete B;
    delete C;
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    std::cout<< "End gemm" << name <<" with time: " <<  elapsed_seconds.count() << std::endl;
}

void Layer::pool () {
    std::cout << name << std::endl;
}

void Layer::upsample(){
    std::cout << name << std::endl;
}

void Layer::forward () {
    if (!type.compare("conv")) 
        conv_gemm();
    else if (!type.compare("pool")) 
        pool();
    else if (!type.compare("upsample")) 
        upsample();
}
