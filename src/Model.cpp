//
// Created by Sergio Martinez on 31/10/2021.
//
#include "ClHelper.h"
#include "Model.h"
#include <algorithm>
#include <utility>
#include <iostream>


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
                     (void *)_weights);
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
    std::cout << exitcode << std::endl;
    exitcode = queue.enqueueReadBuffer(buf_W, CL_TRUE, 0, sizeof(float) * memsize_w, (void *)_weights);
    std::cout << exitcode << std::endl;


    cl::finish();

    for(int i = 0; i<memsize_w; ++i){
        std::cout << *(_weights + i)  << std::endl;
    }
}

Model::Model(float learning_rate, std::vector<int> &dim_sizes) : _learning_rate(learning_rate), _dim_sizes(std::move(dim_sizes)) {
    int memsize = 1;
    for(int _dim_size : _dim_sizes){
        memsize *= _dim_size;
    }

    auto* arr = static_cast<float *>(malloc(memsize * sizeof(float *)));
    for(int i = 0; i < memsize; ++i){
        *(arr+i) = 1;
    }
    _weights = arr;
}

Model::Model(float learning_rate, float* initial_weights, std::vector<int> &dim_sizes) : _learning_rate(learning_rate), _dim_sizes(std::move(dim_sizes))  {
    _weights = initial_weights;
}

float Model::ojas_y(const float* x, int length) {
    float y = 0;
    for(int i = 0; i < length; ++i){
        y+= _weights[i] * (*(x+i));
    }
    return y;
}

const float Model::getLearningRate() const {
    return _learning_rate;
}

const std::vector<int> &Model::getDimSizes() const {
    return _dim_sizes;
}

float *Model::getWeights() const {
    return _weights;
}


