//
// Created by Sergio Martinez on 31/10/2021.
//



#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.hpp>

#else
#include <CL/cl.h>
#endif

cl::Program createProgram( const std::string& file){
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    auto platform = platforms.front();

    assert(platforms.size() > 0);

    std::vector<cl::Device> devices;
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);

    auto device = devices.front();
    assert(devices.size() > 0);

    std::ifstream kernelFile(file);
    std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)), (std::istreambuf_iterator<char>()));

    cl::Program::Sources sources;
    cl::Context context(device);


    sources.push_back({kernelSource.c_str(), kernelSource.length()});

    cl_int exitcode = 0;

    cl::Program program(context, sources, &exitcode);
    program.build();


    assert(exitcode == CL_SUCCESS);

    return program;
}

