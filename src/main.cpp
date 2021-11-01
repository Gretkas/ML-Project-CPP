
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
    std::array<std::array<float, 3>,3> x = {{{0.5,0.5,0.5}, {0.5,0.5,0.5}, {0.5,0.5,0.5}}};
    Model model(0.05, dim);
    for(int i = 0; i<100; ++i){
        model.ojas_rule_openCL((float *)x.data(), 9);
    }

    for(int i = 0; i<9; ++i){
        std::cout << *(model.getWeights() + i)  << std::endl;
    }
    return 0;
}
//

