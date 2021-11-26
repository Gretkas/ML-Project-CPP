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
#include <CL/opencl.hpp>
#include <cassert>

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

    cl::Buffer buf_W(context,
                     CL_MEM_READ_WRITE| CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * _dim_sizes[0],
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
    exitcode = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(_dim_sizes[0]));

    exitcode = queue.enqueueReadBuffer(buf_W, CL_TRUE, 0, sizeof(float) * _dim_sizes[0], (void *)conv_weights);



    cl::finish();
}

Model::Model(float learning_rate, std::vector<int> &dim_sizes) : _learning_rate(learning_rate), _dim_sizes(std::move(dim_sizes)) {
    int memsize = 1;
    for(int _dim_size : _dim_sizes){
        memsize *= _dim_size;
    }
    srand(time(nullptr));
    weights = static_cast<float *>(malloc(memsize * sizeof(float *)));
    for(int i = 0; i < memsize; ++i){
        weights[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    conv_weights = static_cast<float *>(malloc(_dim_sizes[0] * sizeof(float *)));
    for(int i = 0; i < _dim_sizes[0]; ++i){
        conv_weights[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) ;
    }
    initial_pool_weights = static_cast<float *>(malloc(memsize * sizeof(float *)));
    memcpy(initial_pool_weights, weights, sizeof(float) * memsize);

}
float Model::ojas_y(const float* x, int length) {
    float y = 0;
    for(int i = 0; i < length; ++i){
        y+= conv_weights[i] * x[i];
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
    return weights;
}

void Model::decorrelated_hebbian_learning_CPU(float *x, int length) {
    float* y = dhl_y_dot(dhl_y(x, length));
    int memsize_w = 1;
    for(int _dim_size : _dim_sizes){
        memsize_w *= _dim_size;
    }

    for (int i = 0; i < _dim_sizes[0]; ++i) {
        for (int j = 0; j < _dim_sizes[1]; ++j) {
            weights[i*_dim_sizes[0] + j] += y[j]*(x[j] - weights[i*_dim_sizes[0] + j]);
        }
    }
    free(y);
}

Model::~Model() {
    free(weights);
    free(conv_weights);
    free(initial_pool_weights);
}

float* Model::dhl_y(const float *x, int length) {
    auto* exponents = dhl_y_helper_exponent_vector(x, length);
    auto divisor = dhl_y_helper_quotient(exponents);
    auto* y = static_cast<float *>(malloc(length * sizeof(float *)));

    for(int i = 0; i< _dim_sizes[0]; ++i){

        y[i] = exp(exponents[i])/divisor;


    }

    free(exponents);
    return y;
}






float* Model::dhl_y_helper_exponent_vector(const float *x, int length) {
    if(length != _dim_sizes[1] ){
        throw "dimension mismatch";
    }

    auto* exponents = static_cast<float *>(malloc(_dim_sizes[0] * sizeof(float *)));
    for(int i = 0; i<_dim_sizes[0]; ++i){
        exponents[i] = 0;
    }
    for(int i = 0; i< _dim_sizes[0]; ++i){

        for(int j = 0; j<length; ++j){
            exponents[i] += pow(x[j]-weights[i*length+j],2);

        }

        exponents[i] = -exponents[i]/2; //TODO look for ways which allow lower sigma values
    }


    return exponents;
}

float Model::dhl_y_helper_quotient(float *exponents) {
    //TODO find out whether this is worth implementing on GPU, probably not, as it would require an absurdly high first dimension on weights. Ask Ole for clarity on algorithm

    float divisor = 0;
    for(int i = 0; i< _dim_sizes[0]; ++i){

        divisor += exp(exponents[i]); //TODO if sigma is low this becomes 0, is it fixable?

    }

    return divisor;
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
            dw.emplace_back(std::pair<float,int>(abs(initial_pool_weights[_dim_sizes[1]*i + j] - weights[_dim_sizes[1] * i + j]), i));
        }
    }
    //this can probably be done in O(n) but O(nlogn) will have to do
    std::sort(dw.begin(), dw.end(), std::greater<>());
    for(int i = 0; i < n; ++i) {
        best.emplace_back(dw[i].second);
    }
    return best;
}


//found here: https://stackoverflow.com/a/2681094
int previous_power_of_two( int x ) {
    if (x == 0) {
        return 0;
    }
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x - (x >> 1);
}



void Model::dhl_full_gpu(float *x, int len, int num_segments, float sigma) {

    int exitcode;
    cl::Program program = createProgram("Kernels.cl");
    cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
    auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
    auto device = devices.front();

    int memsize_w = 1;
    for(int _dim_size : _dim_sizes){
        memsize_w *= _dim_size;
    }
    auto y = static_cast<float *>(malloc(_dim_sizes[0] * sizeof(float *)));
    for (int i = 0; i < _dim_sizes[0]; ++i) {
        y[i] = 0;
    }
    auto temp_expo = static_cast<float *>(malloc(previous_power_of_two(memsize_w) * sizeof(float *)));
    auto temp_divisor = static_cast<float *>(malloc(previous_power_of_two(_dim_sizes[0]) * sizeof(float *)));
    auto divisor = static_cast<float *>(malloc(sizeof(float *)));
    auto y_sum = static_cast<float *>(malloc(1 * sizeof(float *)));
    auto expo = static_cast<float *>(malloc(memsize_w * sizeof(float *)));

    //initialize all buffers, memory is allocated on the GPU here
    cl::Buffer buf_W(context,
                     CL_MEM_READ_WRITE| CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * memsize_w,
                     (void *)weights);
    cl::Buffer Buf_X(context,
                       CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                       sizeof(float) * len * num_segments,
                       (void *)x);
    cl::Buffer Buf_Y(context,
                     CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * _dim_sizes[0],
                     (void *)y);
    cl::Buffer Buf_vector_sub(context,
                     CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * memsize_w,
                              (void *)expo);
    cl::Buffer Buf_temp_sum(context,
                     CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * previous_power_of_two(_dim_sizes[0]),
                     (void *)temp_divisor);
    cl::Buffer Buf_temp_sum_expo(context,
                            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * previous_power_of_two(memsize_w),
                            (void *)temp_expo);
    cl::Buffer Buf_divisor(context,
                                CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                                sizeof(float),
                                (void *)temp_divisor);
    cl::Buffer Buf_Y_sum(context,
                            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * 1,
                            (void *)y_sum);


    //initialize all kernels
    cl::Kernel kernel_exponent_vector(program, "dhl_y_helper_calc_exponent_vector", &exitcode);
    assert(exitcode == CL_SUCCESS);
    kernel_exponent_vector.setArg(0, Buf_X);
    kernel_exponent_vector.setArg(1, buf_W);
    kernel_exponent_vector.setArg(2, Buf_vector_sub);


    const int offset_1 = previous_power_of_two(len);
    const int surplus_1 = len%previous_power_of_two(len);
    const int wg_size_1 = offset_1;


    cl::Kernel kernel_sum_helper(program, "sum_helper", &exitcode);
    assert(exitcode == CL_SUCCESS);
    kernel_sum_helper.setArg(0, Buf_vector_sub);
    kernel_sum_helper.setArg(1, offset_1);
    kernel_sum_helper.setArg(2, len);
    kernel_sum_helper.setArg(3, Buf_temp_sum_expo);
    kernel_sum_helper.setArg(4, surplus_1);



    cl::Kernel kernel_sum_exponent(program, "sum_reduction", &exitcode);
    assert(exitcode == CL_SUCCESS);

    if(surplus_1 != 0){
        kernel_sum_exponent.setArg(0, Buf_temp_sum_expo);
    }
    else{
        kernel_sum_exponent.setArg(0, Buf_vector_sub);
    }
    kernel_sum_exponent.setArg(1, previous_power_of_two(memsize_w) * sizeof(cl_float), nullptr);
    kernel_sum_exponent.setArg(2, Buf_Y);

    cl::Kernel kernel_y_helper_fraction(program, "dhl_y_helper_fraction", &exitcode);
    assert(exitcode == CL_SUCCESS);
    kernel_y_helper_fraction.setArg(0, Buf_Y);
    kernel_y_helper_fraction.setArg(1, sigma);



    const int offset_2 = previous_power_of_two(_dim_sizes[0]);
    const int surplus_2 = _dim_sizes[0]%previous_power_of_two(_dim_sizes[0]);
    const int wg_size_2 = offset_2;

    cl::Kernel kernel_sum_helper_with_output_divisor(program, "sum_helper_with_output", &exitcode);
    assert(exitcode == CL_SUCCESS);
    kernel_sum_helper_with_output_divisor.setArg(0, Buf_Y);
    kernel_sum_helper_with_output_divisor.setArg(1, Buf_temp_sum);
    kernel_sum_helper_with_output_divisor.setArg(2, offset_2);
    kernel_sum_helper_with_output_divisor.setArg(3, surplus_2);



    cl::Kernel kernel_sum_divisor(program, "sum_reduction", &exitcode);
    assert(exitcode == CL_SUCCESS);
    if(surplus_2 != 0){
        kernel_sum_divisor.setArg(0, Buf_temp_sum);
    }
    else{
        kernel_sum_divisor.setArg(0, Buf_Y);
    }
    kernel_sum_divisor.setArg(1, previous_power_of_two(_dim_sizes[0]) * sizeof(cl_float), nullptr);
    kernel_sum_divisor.setArg(2, Buf_divisor);

    cl::Kernel kernel_y(program, "dhl_y", &exitcode);
    kernel_y.setArg(0, Buf_Y);
    kernel_y.setArg(1, Buf_divisor);


    cl::Kernel kernel_sum_helper_with_output_y_sum(program, "sum_helper_with_power_and_output", &exitcode);
    assert(exitcode == CL_SUCCESS);
    kernel_sum_helper_with_output_y_sum.setArg(0, Buf_Y);
    kernel_sum_helper_with_output_y_sum.setArg(1, Buf_temp_sum);
    kernel_sum_helper_with_output_y_sum.setArg(2, offset_2);
    if(surplus_2 != 0){
        kernel_sum_helper_with_output_y_sum.setArg(3, surplus_2);
    }
    else{
        kernel_sum_helper_with_output_y_sum.setArg(3, offset_2);
    }




    cl::Kernel kernel_sum_y(program, "sum_reduction", &exitcode);
    assert(exitcode == CL_SUCCESS);

    kernel_sum_y.setArg(0, Buf_temp_sum);


    kernel_sum_y.setArg(1, previous_power_of_two(_dim_sizes[0]) * sizeof(cl_float), nullptr);
    kernel_sum_y.setArg(2, Buf_Y_sum);

    cl::Kernel kernel_dhl(program, "decorrelatedHebbianLearning", &exitcode);
    assert(exitcode == CL_SUCCESS);
    kernel_dhl.setArg(0,Buf_X);
    kernel_dhl.setArg(1,buf_W);
    kernel_dhl.setArg(2,Buf_Y);
    kernel_dhl.setArg(3,Buf_Y_sum);
    kernel_dhl.setArg(4,getLearningRate());


    cl::Kernel kernel_reset_y(program, "reset_y", &exitcode);
    assert(exitcode == CL_SUCCESS);
    kernel_reset_y.setArg(0,Buf_Y);
    //command queue for all kernels
    cl::CommandQueue queue(context, device);

    //iterate over all image segments in batch
    //Data will be copied to GPU memory the first time each kernel is called,
    //after which it wont be read untill enqueueReadBuffer is called after the for loop
    for (int i = 0; i < num_segments; ++i) {

        //calulate x-w for the exponents in the y formula
        exitcode = queue.enqueueNDRangeKernel(kernel_exponent_vector, cl::NDRange(len*i), cl::NDRange(memsize_w), cl::NDRange(len));
        assert(exitcode == CL_SUCCESS);
        queue.finish();

        //if size of input is not 2^n it needs to be folded so parallel sum reduction can be used
        if(surplus_1 != 0){
            exitcode = queue.enqueueNDRangeKernel(kernel_sum_helper, cl::NullRange, cl::NDRange(previous_power_of_two(memsize_w)), cl::NDRange(
                    previous_power_of_two(len)));
            assert(exitcode == CL_SUCCESS);
            queue.finish();
        }

        //sum all x-w from last step
        //TODO this only works for small vectors, fix later if time
        exitcode = queue.enqueueNDRangeKernel(kernel_sum_exponent, cl::NullRange, cl::NDRange(previous_power_of_two(len)*_dim_sizes[0]), cl::NDRange(wg_size_1));
        assert(exitcode == CL_SUCCESS);
        queue.finish();


        //exponentiate and divide by sigma
        exitcode = queue.enqueueNDRangeKernel(kernel_y_helper_fraction, cl::NullRange, cl::NDRange(_dim_sizes[0]));
        assert(exitcode == CL_SUCCESS);
        queue.finish();


        //if size of input is not 2^n it needs to be folded so parallel sum reduction can be used
        if(surplus_2 != 0){
            exitcode = queue.enqueueNDRangeKernel(kernel_sum_helper_with_output_divisor, cl::NullRange, cl::NDRange(previous_power_of_two(_dim_sizes[0])));
            assert(exitcode == CL_SUCCESS);
            queue.finish();
        }


        //sum all exponentials from last step to find divisor of y
        //TODO this only works for small vectors, fix later if time
        exitcode = queue.enqueueNDRangeKernel(kernel_sum_divisor, cl::NullRange, cl::NDRange(previous_power_of_two(_dim_sizes[0])), cl::NDRange(wg_size_2));
        assert(exitcode == CL_SUCCESS);
        queue.finish();


        //compute all y values
        exitcode = queue.enqueueNDRangeKernel(kernel_y, cl::NullRange, cl::NDRange(_dim_sizes[0]));
        assert(exitcode == CL_SUCCESS);
        queue.finish();



        // compute all y^2 and fold to a size of 2^n and compute
        exitcode = queue.enqueueNDRangeKernel(kernel_sum_helper_with_output_y_sum, cl::NullRange, cl::NDRange(previous_power_of_two(_dim_sizes[0])));
        assert(exitcode == CL_SUCCESS);
        queue.finish();



        //sum results from last step
        //TODO this only works for small vectors, fix later if time
        exitcode = queue.enqueueNDRangeKernel(kernel_sum_y, cl::NullRange, cl::NDRange(previous_power_of_two(_dim_sizes[0])), cl::NDRange(wg_size_2));
        assert(exitcode == CL_SUCCESS);
        queue.finish();


        //compute weight changes
        exitcode = queue.enqueueNDRangeKernel(kernel_dhl, cl::NDRange(len*i), cl::NDRange(memsize_w), cl::NDRange(len));
        assert(exitcode == CL_SUCCESS);
        queue.finish();


        //reset buffers on GPU so we dont need to read them
        exitcode = queue.enqueueNDRangeKernel(kernel_reset_y, cl::NullRange, cl::NDRange(_dim_sizes[0]));
        assert(exitcode == CL_SUCCESS);
        queue.finish();

    }
    //read weights from GPU and write to model weights
    exitcode = queue.enqueueReadBuffer(buf_W, CL_TRUE, 0, sizeof(float) * memsize_w, (void *)weights);
    queue.finish();
}


