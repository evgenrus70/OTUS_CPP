#include "Layer.h"
#include <functional>
#include <vector>
#include <chrono>
#include <stdlib.h>
#include <omp.h>

Layer::Layer (std::string _type, int _numLayer, int _inFm, int _outFm, int _inSize, int _pad, int _coreSize, int _stride) {
    type = _type;
    numLayer = _numLayer;
    name = type + "_" + std::to_string(numLayer);
    inSize = _inSize;
    inFm = _inFm;
    outFm = _outFm;
    coreSize = _coreSize;
    pad = _pad;
    stride = _stride;
}

void Layer::print(){
    std::cout << name <<std::endl;
}

void Layer::printInputs(){
    std::cout <<"layer " << numLayer << " inputs" << std::endl;
    for (int i = 0; i < inSize; i++) {
        for (int j = 0; j < inSize; j++) {
            for (int k = 0; k < inFm; k++) {
                std::cout << inputData[inSize*inSize*k + inSize*j + i];
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
                std::cout << outputData[inSize*inSize*k + inSize*j + i];
            }
        }
    }
    std::cout<<"\n";
}

void Layer::im2col (float* data_im, float* data_col) {
    int c,h,w;
    int inSize_true = inSize - 2 * pad;
    int height_col = (inSize_true + 2*pad - coreSize) / stride + 1;
    int width_col = (inSize_true + 2*pad - coreSize) / stride + 1;

    int channels_col = inFm * coreSize * coreSize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % coreSize;
        int h_offset = (c / coreSize) % coreSize;
        int c_im = c / coreSize / coreSize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2colGetPixel(data_im, inSize_true, inSize_true, inFm, im_row, im_col, c_im, pad);
            }
        }
    }
}

float Layer::im2colGetPixel (float *im, int height, int width, int channels, int row, int col, int channel, int pad){
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) 
        return 0;
    return im[col + width*(row + height*channel)];
}

void Layer::conv () {
    std::cout<<"Start " << name << std::endl;
    const auto start{std::chrono::steady_clock::now()};
    int inSize_true = inSize - 2 * pad;
    int outSize_true = (inSize_true - coreSize) / stride + 1;
    int outSize = outSize_true + 2 * pad;
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

    weights = new float[outFm * inFm * coreSize * coreSize];            // weights
    float* inputs = new float[N1 * N1 * coreSize * coreSize * inFm];    // input data
    outputData = new float[N1 * N1 * outFm];                            // result
    float *true_inputs = NULL; 

    if (pad){
        true_inputs = new float[inFm * inSize_true* inSize_true];
        for (int c = 0, true_c = 0; c < inFm; ++c,++true_c){
            for (int y = pad, true_y = 0; y < inSize-pad; ++y, ++true_y) {
                for (int x = pad, true_x =0; x < inSize-pad; ++x,++true_x) {
                    true_inputs[true_c * inSize_true * inSize_true + true_y*inSize_true + true_x ] = inputData[c*inSize*inSize + y*inSize + x];
                }
            }
        }
    } else {
        true_inputs = inputData;
    }

    if (coreSize == 1) {
        inputs = true_inputs;
    } else {
        im2col(true_inputs,inputs);
    }

    int i,j,j1,j2,k;
    //#pragma omp parallel for num_threads(4)
    for(i = M_start; i < M_start + M; ++i){   // i = 0; i < 64; i++
        for(k = 0; k < K; ++k){             // k = 0; k < 27; ++k
            j = 0;                          
            float A_PART = weights[i*lda+k]; // A_PART = A[0*27+0] : A[63*27+26];
            for(j1 = pad_t; j1 < N1; ++j1) {    // j1 = 1; j1 < 258; ++j1
                for(j2 = pad_l; j2 < N2; ++j2) {// j2 = 1; j2 < 258; ++j2
                    outputData[i*ldc1 + j1*ldc2 + j2] += A_PART*inputs[k*ldb + j]; // C[0*65536 + 1*256 + 1] += A_PART*B[0*65536 + 0] : C[63*65536 + 257*256 + 257]
                    j++;
                }
            }
        }
    }
    
    delete weights;
    delete inputs;
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    std::cout<< "End gemm " << name <<" with time: " <<  elapsed_seconds.count() << std::endl;
}

void Layer::pool () {
    std::cout<<"Start " << name << std::endl;
    const auto start{std::chrono::steady_clock::now()};
    int inSize_true = inSize - 2 * pad;
    int outSize_true = (inSize_true - coreSize) / stride + 1;
    int outSize = outSize_true + 2 * pad;
    int tmp_max        = 0;
    int input_x        = 0;
    int input_y        = 0;
    int output_c       = 0;
    int prev_input_x   = 0;
    int prev_input_y   = 0;
    int weight_shift_x = 0;
    int weight_shift_y = 0;
    int zero_fr_left   = pad;
    int zero_fr_top    = pad;
    int output_x       = zero_fr_left;
    int output_y       = zero_fr_top;
    int prev_output_x  = zero_fr_left;
    int prev_output_y  = zero_fr_top;
    int stride_x       = stride;
    int output_w_calc  = outSize_true + pad ;
    int output_h_calc  = outSize_true + pad;

    outputData = new float[outSize * outSize * outFm];
    float *true_inputs = new float[inFm * inSize_true * inSize_true];
    for (int c = 0, true_c = 0; c < inFm; ++c,++true_c){
        for (int y = pad, true_y = 0; y < inSize-pad; ++y, ++true_y) {
            for (int x = pad, true_x = 0; x < inSize-pad; ++x,++true_x) {
                true_inputs[true_c * inSize_true * inSize_true + true_y*inSize_true + true_x ] = inputData[c*inSize*inSize + y*inSize + x];
            }
        }
    }
    /*
    size_type pos_in_3d_plane = x_ * y_ * z_pos;
    size_type pos_in_2d_plane = x_ * y_pos;
    size_type pos_in_1d_plane = x_pos;
    return data_[pos_in_3d_plane + pos_in_2d_plane + pos_in_1d_plane];
    */

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
                    outputData[outFm * outSize * output_x + outFm * output_y + output_x] = tmp_max;
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
    delete true_inputs;
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    std::cout<< "End " << name <<" with time: " <<  elapsed_seconds.count() << std::endl;
}

void Layer::upsample(){
    std::cout<<"Start " << name << std::endl;
    const auto start{std::chrono::steady_clock::now()};
    int inSize_true = inSize - 2 * pad;
    int outSize_true = inSize_true * coreSize;
    int outSize = outSize_true + 2 * pad;
    int tmp_max        = 0;
    int input_x        = 0;
    int input_y        = 0;
    int output_c       = 0;
    int prev_input_x   = 0;
    int prev_input_y   = 0;
    int weight_shift_x = 0;
    int weight_shift_y = 0;
    int zero_fr_left   = pad;
    int zero_fr_top    = pad;
    int output_x       = zero_fr_left;
    int output_y       = zero_fr_top;
    int prev_output_x  = zero_fr_left;
    int prev_output_y  = zero_fr_top;
    int stride_x       = stride;
    int output_w_calc  = outSize_true + pad ;
    int output_h_calc  = outSize_true + pad;

    outputData = new float[outSize * outSize * outFm];
    float *true_inputs = new float[inFm * inSize_true * inSize_true];

    for (int c = 0, true_c = 0; c < inFm; ++c,++true_c){
        for (int y = pad, true_y = 0; y < inSize - pad; ++y, ++true_y) {
            for (int x = pad, true_x =0; x < inSize - pad; ++x,++true_x) {
                true_inputs[true_c * inSize_true * inSize_true + true_y*inSize_true + true_x ] = inputData[c*inSize*inSize + y*inSize + x];
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
                            outputData[outFm * outSize * output_x + outFm * output_y + output_c] = true_inputs[outFm * inSize_true * input_x + outFm * input_y + output_c];
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
    delete true_inputs;
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    std::cout<< "End " << name <<" with time: " <<  elapsed_seconds.count() << std::endl;
}

void Layer::addBias () {
    std::cout<<"Add bias to " << name << std::endl;
    int inSize_true = inSize - 2 * pad;
    int outSize_true = inSize_true ;
    int outSize = outSize_true + 2 * pad;
    int n = outSize * outSize;
    int m = outFm;

    int n1 = outSize_true + pad;
    int n2 = outSize_true + pad;
    int calc_size = outSize_true * outSize_true;

    float *biases_ptr = new float[outFm];
    outputData = new float[outSize * outSize * outFm];

    int i,j1,j2,k;
    //#pragma omp parallel for
    for(i = 0; i < m; ++i){
        float A_PART = biases_ptr[i];
        for(j1 = pad; j1 < n1; ++j1){
            for(j2 = pad; j2 < n2; ++j2){
                outputData[i*n+ j1*outSize + j2] += A_PART;
            }
        }
    }
    delete biases_ptr;
}

void Layer::activate(){
    std::cout<<"Activation function of " << name << std::endl;
    int outSize =  inSize * inSize * outFm;
    float neg_coef = 0.3f;
    float pos_coef = 1;
    for (int i = 0; i < outSize;++i){
        outputData[i] = outputData[i] > 0 ? outputData[i]*pos_coef : outputData[i]*neg_coef;
    }
}

void Layer::normalize(){
    std::cout<<"Normalization of " << name << std::endl;
    float *alpha_coefs = new float[inFm];
    float *beta_coefs = new float[inFm];
    std::cout << "inW: " << inSize << std::endl;
    for(int i=0; i < inFm; ++i){
        for (int j = pad; j < inSize - pad; ++j){
            for (int k = pad; k < inSize - pad; ++k){
               outputData[inFm*inSize*k + inFm*j + i] = outputData[inFm*inSize*k + inFm*j + i] * alpha_coefs[i] + beta_coefs[i];
            }
        }
    }
    delete alpha_coefs;
    delete beta_coefs;
}

void Layer::postProcessing () {
    addBias();
    activate();
    normalize();
}

void Layer::forward () {
    if (!type.compare("conv")) {
        conv();
        postProcessing();
    }  
    else if (!type.compare("pool")) 
        pool();
    else if (!type.compare("upsample")) 
        upsample();
}
