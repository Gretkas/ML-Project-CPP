
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
    std::array<std::array<float, 3>,3> x = {{{2.0,2.0,2.0}, {2.0,2.0,2.0}, {2.0,2.0,2.0}}};
    Model model(0.05, dim);

    model.ojas_rule_openCL((float *)x.data(), 9);
    return 0;
}
//

