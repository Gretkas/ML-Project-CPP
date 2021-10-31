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
    assert(exitcode == CL_SUCCESS);

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


/*int main() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    assert(platforms.size() > 0);

    auto platform = platforms.front();

    std::vector<cl::Device> devices;
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);

    assert(devices.size() > 0);

    auto device = devices.front();
    auto vendor = device.getInfo<CL_DEVICE_VENDOR>();
    auto version = device.getInfo<CL_DEVICE_VERSION>();

    std::cout << "Device Vendor: " << vendor << std::endl;
    std::cout << "Device Version: " << version << std::endl;

    cl::Context context(device);
    std::ifstream kernelFile("Kernels.cl");
    std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)), (std::istreambuf_iterator<char>()));

    cl::Program::Sources sources;



    sources.push_back({kernelSource.c_str(), kernelSource.length()});

    cl_int exitcode = 0;

    cl::Program program(context, sources, &exitcode);
    program.build();
    assert(exitcode == CL_SUCCESS);

    cl::Kernel kernel(program, "squareArray", &exitcode);
    assert(exitcode == CL_SUCCESS);

    auto workGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
    std::cout << "Kernel Work Group Size: " << workGroupSize << std::endl;

    std::vector<int> outVec(4);
    std::vector<int> inVec(4);
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
}*/
