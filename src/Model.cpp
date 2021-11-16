//
// Created by Sergio Martinez on 31/10/2021.
//
#include "ClHelper.h"
#include "Model.h"
#include <algorithm>
#include <utility>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.hpp>
#include <fstream>

#else
#include <CL/cl.h>
#endif




void Model::ojas_rule_openCL(float* x, int length) {
    const float y = ojas_y(x, length);
    int exitcode;
    cl::Program program = createProgram("Kernels.cl");
    cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
    auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
    auto device = devices.front();
    cl::Kernel kernel(program, "ojasRule", &exitcode);
    assert(exitcode == CL_SUCCESS);

    int memsize_w = 1;

    for(int _dim_size : _dim_sizes){
        memsize_w *= _dim_size;
    }

    cl::Buffer buf_W(context,
                     CL_MEM_READ_WRITE| CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * memsize_w,
                     (void *)conv_weights);
    cl::Buffer inBuf_X(context,
                       CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * length,
                       (void *)x);


    kernel.setArg(0, inBuf_X);
    kernel.setArg(1, buf_W);
    kernel.setArg(2, y);
    kernel.setArg(3,_learning_rate);


    cl::CommandQueue queue(context, device);
    exitcode = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(memsize_w));
    //std::cout << exitcode << std::endl;
    exitcode = queue.enqueueReadBuffer(buf_W, CL_TRUE, 0, sizeof(float) * memsize_w, (void *)conv_weights);
    //std::cout << exitcode << std::endl;


    cl::finish();
}

Model::Model(float learning_rate, std::vector<int> &dim_sizes) : _learning_rate(learning_rate), _dim_sizes(std::move(dim_sizes)) {
    int memsize = 1;
    for(int _dim_size : _dim_sizes){
        memsize *= _dim_size;
    }

    auto* pool_arr = static_cast<float *>(malloc(memsize * sizeof(float *)));
    srand(time(nullptr));
    for(int i = 0; i < memsize; ++i){
        pool_arr[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    auto* conv_arr = static_cast<float *>(malloc(memsize * sizeof(float *)));
    srand(time(nullptr));
    for(int i = 0; i < memsize; ++i){
        conv_arr[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    pool_weights = pool_arr;
    memcpy(initial_pool_weights, pool_weights, sizeof(float)*memsize);
    conv_weights = conv_arr;

}
float Model::ojas_y(const float* x, int length) {
    float y = 0;
    for(int i = 0; i < length; ++i){
        y+= conv_weights[i] * (*(x + i));
    }
    return y;
}

const float Model::getLearningRate() const {
    return _learning_rate;
}

const std::vector<int> &Model::getDimSizes() const {
    return _dim_sizes;
}

float *Model::getConvWeights() const {
    return conv_weights;
}

float *Model::getPoolWeights() const {
    return pool_weights;
}

void Model::decorrelated_hebbian_learning_openCL(float *x, int length) {
    float* y = dhl_y_dot(dhl_y(x, length));
    int exitcode;
    cl::Program program = createProgram("Kernels.cl");
    cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
    auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
    auto device = devices.front();
    cl::Kernel kernel(program, "decorrelatedHebbianLearning", &exitcode);
    assert(exitcode == CL_SUCCESS);

    int memsize_w = 1;
    for(int _dim_size : _dim_sizes){
        memsize_w *= _dim_size;
    }

    cl::Buffer buf_W(context,
                     CL_MEM_READ_WRITE| CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * memsize_w,
                     (void *)pool_weights);
    cl::Buffer inBuf_X(context,
                       CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * length,
                       (void *)x);
    cl::Buffer inBuf_Y(context,
                       CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * _dim_sizes[0],
                       (void *)y);


    kernel.setArg(0, inBuf_X);
    kernel.setArg(1, buf_W);
    kernel.setArg(2, inBuf_Y);
    kernel.setArg(3,_dim_sizes[0]);
    kernel.setArg(4,length);


    cl::CommandQueue queue(context, device);
    exitcode = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(memsize_w));
    exitcode = queue.enqueueReadBuffer(buf_W, CL_TRUE, 0, sizeof(float) * memsize_w, (void *)pool_weights);
    cl::finish();
    free(y);
}

Model::~Model() {
    free(pool_weights);
}

float* Model::dhl_y(const float *x, int length) {
    auto* exponents = dhl_y_helper_exponent_vector(x, length);
    auto quotient = dhl_y_helper_quotient(exponents);
    auto* y = static_cast<float *>(malloc(length * sizeof(float *)));

    for(int i = 0; i< _dim_sizes[0]; ++i){
        y[i] = exponents[i]/quotient;
    }

    free(exponents);
    return y;
}






float* Model::dhl_y_helper_exponent_vector(const float *x, int length) {
    if(length != _dim_sizes[1] ){
        throw "dimension mismatch";
    }


    int exitcode;
    cl::Program program = createProgram("Kernels.cl");
    cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
    auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
    auto device = devices.front();
    cl::Kernel kernel_vector(program, "dhl_y_helper_calc_exponent_vector", &exitcode);
    assert(exitcode == CL_SUCCESS);
    int memsize_w = 1;
    for(int _dim_size : _dim_sizes){
        memsize_w *= _dim_size;
    }

    auto* outvec = static_cast<float *>(malloc(memsize_w * sizeof(float* )));
    cl::Buffer buf_W(context,
                     CL_MEM_READ_ONLY| CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * memsize_w,
                     (void *)pool_weights);
    cl::Buffer inBuf_X(context,
                       CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * length,
                       (void *)x);
    cl::Buffer outbuf(context,
                      CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof(float) * memsize_w,
                      (void *)outvec);


    kernel_vector.setArg(0, inBuf_X);
    kernel_vector.setArg(1, buf_W);
    kernel_vector.setArg(2, outbuf);
    kernel_vector.setArg(3, length);
    cl::CommandQueue queue(context, device);
    exitcode = queue.enqueueNDRangeKernel(kernel_vector, cl::NullRange, cl::NDRange(memsize_w));
    exitcode = queue.enqueueReadBuffer(outbuf, CL_TRUE, 0, sizeof(float) * memsize_w, (void *)outvec);
    cl::finish();


    // outvec has same dimentions as w at this point, needs to be reduced to _dim_size[0]

    //TODO implement this on GPU, hard for input that doesnt have size of 2^n, and slower for x length * _dim_sizes[0] < 10^6
    //TODO if quotient is worth implementing on GPU, pad exponents with 0s to allow for parallel sum reduction
    auto* exponents = static_cast<float *>(malloc(_dim_sizes[0] * sizeof(float *)));
    for(int i = 0; i<_dim_sizes[0]; ++i){
        exponents[i] = 0;
    }
    for(int i = 0; i< _dim_sizes[0]; ++i){

        for(int j = 0; j<length; ++j){

            exponents[i] += -pow(outvec[i*length+j],2);

        }

        exponents[i] = exponents[i]/25; //TODO look for ways which allow lower sigma values

    }
    free(outvec);


    return exponents;
}

float Model::dhl_y_helper_quotient(float *exponents) {
    //TODO find out whether this is worth implementing on GPU, probably not, as it would require an absurdly high first dimension on weights. Ask Ole for clarity on algorithm

    float quotient = 0;
    for(int i = 0; i< _dim_sizes[0]; ++i){

        quotient += exp(exponents[i]); //TODO if sigma is low this becomes 0, is it fixable?

    }

    return quotient;
}
//TODO might also be worth running on GPU if sufficiently large weights
//calculates  (y - sum y^2) * learning_rate * y
float* Model::dhl_y_dot(float *y) {
    float y_dot = 0;
    for(int i = 0; i < _dim_sizes[0]; ++i){
        y_dot += y[i] * y[i];    //TODO if sigma is low this becomes inf, is it fixable?
    }

    for(int i = 0; i < _dim_sizes[0]; ++i){

        y[i] = y[i]*_learning_rate * (y[i] - y_dot);

    }

    return y;
}

//returns indices of best filters
std::vector<int> Model::find_active(int n){
    std::vector<int> best;
    best.reserve(n);
    std::vector<std::pair<float, int>> dw;
    for (int i = 0; i < _dim_sizes[0]; ++i) {
        for (int j = 0; j < _dim_sizes[1]; ++j) {
            dw.emplace_back(std::pair<float,int>(abs(initial_pool_weights[_dim_sizes[1]*i + j] - pool_weights[_dim_sizes[1]*i + j]), i));
        }
    }
    //this can probably be done in O(n) but O(nlogn) will have to do
    std::sort(dw.begin(), dw.end(), std::greater<>());
    for(int i = 0; i < n; ++i) {
        best.emplace_back(dw[i].second);
    }
    return best;
}
