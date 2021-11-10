
#include "ClHelper.h"
#include "Model.h"
#include <iostream>
#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.hpp>
#include "MNIST/mnist_loader.h"
#include <fstream>

#else
#include <CL/cl.h>
#endif


void printweights(const Model& m){
    int size = m.getDimSizes()[1];
    for(int j = 0; j < m.getDimSizes()[0]; ++j){
        std::string out;
        for(int i = 0; i < size; ++i){
            if(i%25 == 24){
                out += std::to_string(m.getWeights()[25*j+i]);
                std::cout << out << std::endl;
                out = "";
            }
            else{
                out += std::to_string(m.getWeights()[25*j+i]) + " ";
            }
        }
    }

}


int main() {

    mnist_loader train("dataset/train-images-idx3-ubyte",
                       "dataset/train-labels-idx1-ubyte", 100);
    mnist_loader test("dataset/t10k-images-idx3-ubyte",
                      "dataset/t10k-labels-idx1-ubyte", 100);
    //std::vector<float> image = train.images(0);

    //std::array<std::array<float, 3>,3> x = {{{0.5,0.5,0.5}, {0.5,0.5,0.5}, {0.5,0.5,0.5}}};
    std::vector<int> dim = {25, 25};
    //use numbers between 0 and 1 or bad things happen
    Model model(1, dim);
    printweights(model);

    for(int i = 0; i<100; ++i){
        model.decorrelated_hebbian_learning_openCL(train.image_segment().data(), 25);
        printweights(model);
    }


    return 0;
}
//

