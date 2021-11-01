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




void Model::ojas_rule_openCL(std::vector<float> x) {
    const float y = ojas_y(x);

    int exitcode;
    cl::Program program = createProgram("Kernels.cl");
    cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
    auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
    auto device = devices.front();
    cl::Kernel kernel(program, "ojasRule", &exitcode);
    assert(exitcode == CL_SUCCESS);


    cl::Buffer buf_W(context,
                     CL_MEM_READ_WRITE| CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * _weights.size(),
                     _weights.data());
    cl::Buffer inBuf_X(context,
                       CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * x.size(),
                       x.data());


    kernel.setArg(0, inBuf_X);
    kernel.setArg(1, buf_W);
    kernel.setArg(2, y);
    kernel.setArg(3,_learning_rate);


    cl::CommandQueue queue(context, device);
    exitcode = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(_weights.size()));
    std::cout << exitcode << std::endl;
    exitcode = queue.enqueueReadBuffer(buf_W, CL_TRUE, 0, sizeof(float) * _weights.size(), _weights.data());
    std::cout << exitcode << std::endl;

    cl::finish();

    for(int i = 0; i<3; ++i){
        std::cout << _weights[i] << std::endl;
    }
}

Model::Model(float learning_rate, std::vector<int> &dim_sizes) : _learning_rate(learning_rate), _dim_sizes(std::move(dim_sizes)) {
    int memsize = 1;
    for(int i = 0; i<_dim_sizes.size(); ++i){
        memsize *= _dim_sizes[i];
    }
    _weights.reserve(memsize);
    std::fill(_weights.begin(), _weights.end(), 1);
}

Model::Model(float learning_rate, std::vector<float> initial_weights, std::vector<int> &dim_sizes) : _learning_rate(learning_rate), _dim_sizes(std::move(dim_sizes))  {
    _weights = std::move(initial_weights);
}

float Model::ojas_y(std::vector<float> x) {
    float y;
    for(int i = 0; i < _weights.size(); ++i){
        y+= _weights[i] * x[i];
    }
    return y;
}

const std::vector<float> &Model::getWeights() const {
    return _weights;
}

Model::Model(float learning_rate, std::vector<float> initial_weights) : _learning_rate(learning_rate),  _dim_sizes(1,(int) initial_weights.size())  {
    _weights = std::move(initial_weights);
}
