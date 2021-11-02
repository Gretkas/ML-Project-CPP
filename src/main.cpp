
#include "ClHelper.h"
#include "Model.h"
#include <iostream>
#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.hpp>
#include <fstream>

#else
#include <CL/cl.h>
#endif
int main() {
    std::vector<int> dim(2,3);
    //use numbers between 0 and 1 or bad things happen
    Model model(0.05, dim);
    std::array<float,3> x = {0.5,0.5,0.5};
    for(int i = 0; i<20; ++i){
        model.decorrelated_hebbian_learning_openCL((float *)x.data(), 3);
    }

    for(int i = 0; i<9; ++i){
        std::cout << *(model.getWeights() + i)  << std::endl;
    }
    return 0;
}
//    std::array<std::array<float, 3>,3> x = {{{0.5,0.5,0.5}, {0.5,0.5,0.5}, {0.5,0.5,0.5}}};

