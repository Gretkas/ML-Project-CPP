//
// Created by Sergio Martinez on 31/10/2021.
//

#include <string>
#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.h>
#endif

#ifndef ML_PROJECT_CPP_CLHELPER_H
#define ML_PROJECT_CPP_CLHELPER_H

#endif //ML_PROJECT_CPP_CLHELPER_H
cl::Program createProgram(const std::string& file);

