#include "ClHelper.h"
#include <iostream>
#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.hpp>
#include <fstream>

#else
#include <CL/cl.h>
#endif
int main() {
    int exitcode;
    cl::Program program = createProgram("Kernels.cl");
    cl::Context context = program.getInfo<CL_PROGRAM_CONTEXT>();
    auto devices = program.getInfo<CL_PROGRAM_DEVICES>();
    auto device = devices.front();
    cl::Kernel kernel(program, "squareArray", &exitcode);
    std::cout << exitcode << std::endl;
    //assert(exitcode == CL_SUCCESS);

    std::vector<int> outVec(3);
    std::vector<int> inVec(3);
    std::fill(inVec.begin(), inVec.end(), 2);

    cl::Buffer inBuf(context,
                     CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * inVec.size(),
                     inVec.data());
    cl::Buffer outBuf(context,
                      CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                      sizeof(int) * outVec.size());
    kernel.setArg(0, inBuf);
    kernel.setArg(1, outBuf);

    cl::CommandQueue queue(context, device);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inVec.size()));
    queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(int) * outVec.size(), outVec.data());

    for (std::vector<int>::const_iterator i = outVec.begin(); i != outVec.end(); ++i)
        std::cout << *i << std::endl;

    return 0;
}



