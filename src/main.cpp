
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
    std::vector<float> w(3,1);
    std::vector<float> x(3,2);
    Model model(0.05, w);
    model.ojas_rule_openCL(x);

    for(int i = 0; i<3; ++i){
        std::cout << model.getWeights()[i] << std::endl;
    }
    return 0;
}


