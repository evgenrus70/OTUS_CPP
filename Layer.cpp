#include "Layer.h"
#include <functional>
#include <vector>
#include <chrono>
#include <stdlib.h>
#include <omp.h>

Layer::Layer (std::string _type, int _numLayer, int _inFm, int _outFm, int _inSize, int _border, int _coreSize, int _stride) {
    type = _type;
    numLayer = _numLayer;
    name = type + "_" + std::to_string(numLayer);
    inSize = _inSize;
    inFm = _inFm;
    outFm = _outFm;
    coreSize = _coreSize;
    border = _border;
    stride = _stride;
    //inputData = vector_3d(inSize,inSize,inFm);
    //outputData = vector_3d(inSize,inSize,outFm);

    vector_3d<int> _inputData (inSize,inSize,inFm);
    vector_3d<int> _outputData (inSize,inSize,outFm);
    vector_4d<int> _weights (inFm,outFm,coreSize,coreSize);
    inputData = _inputData;
    outputData = _outputData;
    weights = _weights;
}

void Layer::print(){
    std::cout << name <<std::endl;
}

void Layer::printInputs(){
    std::cout <<"layer " << numLayer << " inputs" << std::endl;
    for (int i = 0; i < inSize; i++) {
        for (int j = 0; j < inSize; j++) {
            for (int k = 0; k < inFm; k++) {
                std::cout << inputData(i,j,k);
            }
        }
    }    
    std::cout<<"\n";
}

void Layer::printOutputs(){
    std::cout <<"layer " << numLayer << " outputs" << std::endl;
    for (int i = 0; i < inSize; i++) {
        for (int j = 0; j < inSize; j++) {
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
    vector_4d<int> out_tmp (inSize,inSize,outFm,inFm); 
    int i = 0, j = 0;
    for (int ix = 0; ix < outFm; ix++) {
        for (int iy = 0; iy < inFm; iy++) {
            for (int x = 0; x < inSize - 2*border; x++) {
                for (int y = 0; y < inSize - 2*border; y++) {
                    for (i = 0; i < coreSize; i++) {
                        for (j = 0; j < coreSize; j++) {
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
    int inSize_true = inSize - 2 * border;
    int outSize_true = (inSize_true - coreSize) / stride + 1;
    int outSize = outSize_true + 2 * border;
    int M = outFm;
    int N1 = inSize;
    int N2 = inSize;
    int K = coreSize * coreSize * inFm;
    int lda = K;
    int ldb = outSize_true * outSize_true;
    int ldc1 = outSize * outSize;
    int M_start = 0;
    int pad_t = stride;
    int pad_l = stride;
    int ldc2 = outSize;

    float *A = new float[outFm * inFm * coreSize * coreSize];       // weights
    float *B = new float[N1 * N1 * coreSize * coreSize * inFm];     // input data
    float *C = new float[N1 * N1 * outFm];                          // result

    int i,j,j1,j2,k;
    //#pragma omp parallel for num_threads(4)
    for(i = M_start; i < M_start + M; ++i){   // i = 0; i < 64; i++
        for(k = 0; k < K; ++k){             // k = 0; k < 27; ++k
            j = 0;                          
            float A_PART = A[i*lda+k]; // A_PART = A[0*27+0] : A[63*27+26];
            for(j1 = pad_t; j1 < N1; ++j1) {    // j1 = 1; j1 < 258; ++j1
                for(j2 = pad_l; j2 < N2; ++j2) {// j2 = 1; j2 < 258; ++j2
                    C[i*ldc1 + j1*ldc2 + j2] += A_PART*B[k*ldb + j]; // C[0*65536 + 1*256 + 1] += A_PART*B[0*65536 + 0] : C[63*65536 + 257*256 + 257]
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
    std::cout<< "End gemm " << name <<" with time: " <<  elapsed_seconds.count() << std::endl;
}

void Layer::pool () {
    std::cout<<"Start " << name << std::endl;
    const auto start{std::chrono::steady_clock::now()};
    int inSize_true = inSize - 2 * border;
    int outSize_true = (inSize_true - coreSize) / stride + 1;
    int outSize = outSize_true + 2 * border;
    int tmp_max        = 0;
    int input_x        = 0;
    int input_y        = 0;
    int output_c       = 0;
    int prev_input_x   = 0;
    int prev_input_y   = 0;
    int weight_shift_x = 0;
    int weight_shift_y = 0;
    int zero_fr_left   = border;
    int zero_fr_top    = border;
    int output_x       = zero_fr_left;
    int output_y       = zero_fr_top;
    int prev_output_x  = zero_fr_left;
    int prev_output_y  = zero_fr_top;
    int stride_x       = stride;
    int output_w_calc  = outSize_true + border ;
    int output_h_calc  = outSize_true + border;

    float *inputs_ptr = new float[inSize * inSize * coreSize * coreSize * inFm];
    float *outputs_ptr = new float[outSize * outSize * outFm];
    float *true_inputs = new float[inFm * inSize_true * inSize_true];
    for (int c = 0, true_c = 0; c < inFm; ++c,++true_c){
        for (int y = border, true_y = 0; y < inSize-border; ++y, ++true_y) {
            for (int x = border, true_x = 0; x < inSize-border; ++x,++true_x) {
                true_inputs[true_c * inSize_true * inSize_true + true_y*inSize_true + true_x ] = inputs_ptr[c*inSize*inSize + y*inSize + x];
            }
        }
    }
    /*
    size_type pos_in_3d_plane = x_ * y_ * z_pos;
    size_type pos_in_2d_plane = x_ * y_pos;
    size_type pos_in_1d_plane = x_pos;
    return data_[pos_in_3d_plane + pos_in_2d_plane + pos_in_1d_plane];
    */
    //float (*inputs_data)[256][256];
    //inputs_data = ( float(*)[256][256])true_inputs;

    //float(*outputs_data)[258][258];
    //outputs_data = ( float(*)[258][258])outputs_ptr;

    for (output_y = zero_fr_top ; output_y < output_h_calc; ++output_y) {
            for (output_x = zero_fr_left ; output_x < output_w_calc; ++output_x) {
                for (output_c = 0; output_c < outFm; ++output_c) {
                    tmp_max = true_inputs[outFm * inSize_true * input_x + outFm * input_y + output_c];
                    for (weight_shift_y = 0; weight_shift_y < coreSize; ++weight_shift_y) {
                        for (weight_shift_x = 0; weight_shift_x < coreSize; ++weight_shift_x) {
                            if (tmp_max < true_inputs[outFm * inSize_true * input_x + outFm * input_y + output_c]){
                                tmp_max = true_inputs[outFm * inSize_true * input_x + outFm * input_y + output_c];
                            }
                            ++input_x;
                        }
                        input_x = prev_input_x;
                        ++input_y;
                    }
                    outputs_ptr[outFm * outSize * output_x + outFm * output_y + output_x] = tmp_max;
                    input_x = prev_input_x;
                    input_y = prev_input_y;
                }
                prev_input_x += stride_x;
                input_y = prev_input_y;
                input_x = prev_input_x;
            }
            prev_input_x = 0;
            input_x = 0;
            prev_input_y += stride_x;
            input_y = prev_input_y;
    }
    delete inputs_ptr;
    delete outputs_ptr;
    delete true_inputs;
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    std::cout<< "End " << name <<" with time: " <<  elapsed_seconds.count() << std::endl;
}

void Layer::upsample(){
    std::cout<<"Start " << name << std::endl;
    const auto start{std::chrono::steady_clock::now()};
    int inSize_true = inSize - 2 * border;
    int outSize_true = inSize_true * coreSize;
    int outSize = outSize_true + 2 * border;
    int tmp_max        = 0;
    int input_x        = 0;
    int input_y        = 0;
    int output_c       = 0;
    int prev_input_x   = 0;
    int prev_input_y   = 0;
    int weight_shift_x = 0;
    int weight_shift_y = 0;
    int zero_fr_left   = border;
    int zero_fr_top    = border;
    int output_x       = zero_fr_left;
    int output_y       = zero_fr_top;
    int prev_output_x  = zero_fr_left;
    int prev_output_y  = zero_fr_top;
    int stride_x       = stride;
    int output_w_calc  = outSize_true + border ;
    int output_h_calc  = outSize_true + border;

    float *inputs_ptr = new float[inSize * inSize * coreSize * coreSize * inFm];
    float *outputs_ptr = new float[outSize * outSize * outFm];
    float *true_inputs = new float[inFm * inSize_true * inSize_true];

    for (int c = 0, true_c = 0; c < inFm; ++c,++true_c){
        for (int y = border, true_y = 0; y < inSize - border; ++y, ++true_y) {
            for (int x = border, true_x =0; x < inSize - border; ++x,++true_x) {
                true_inputs[true_c * inSize_true * inSize_true + true_y*inSize_true + true_x ] = inputs_ptr[c*inSize*inSize + y*inSize + x];
            }
        }
    }

      /*
    size_type pos_in_3d_plane = x_ * y_ * z_pos;
    size_type pos_in_2d_plane = x_ * y_pos;
    size_type pos_in_1d_plane = x_pos;
    return data_[pos_in_3d_plane + pos_in_2d_plane + pos_in_1d_plane];
    */
    for (input_y = 0 ; input_y < inSize_true; ++input_y) {
            for (input_x = 0 ; input_x < inSize_true; ++input_x) {
                for (output_c = 0; output_c < outFm; ++output_c) {
                    tmp_max = true_inputs[outFm * inSize_true * input_x + outFm * input_y + output_c];
                    for (weight_shift_y = 0; weight_shift_y < coreSize; ++weight_shift_y) {
                        for (weight_shift_x = 0; weight_shift_x < coreSize; ++weight_shift_x) {
                            outputs_ptr[outFm * outSize * output_x + outFm * output_y + output_c] = true_inputs[outFm * inSize_true * input_x + outFm * input_y + output_c];
                            ++output_x;
                        }
                        output_x = prev_output_x;
                        ++output_y;
                    }
                    output_x = prev_output_x;
                    output_y = prev_output_y;
                }
                prev_output_x += stride_x;
                output_y       = prev_output_y;
                output_x       = prev_output_x;
            }
            prev_output_x  = zero_fr_left;
            output_x       = zero_fr_left;
            prev_output_y += stride_x;
            output_y       = prev_output_y;
    }
    delete inputs_ptr;
    delete outputs_ptr;
    delete true_inputs;
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    std::cout<< "End " << name <<" with time: " <<  elapsed_seconds.count() << std::endl;
}

void Layer::forward () {
    if (!type.compare("conv")) 
        conv_gemm();
    else if (!type.compare("pool")) 
        pool();
    else if (!type.compare("upsample")) 
        upsample();
}
