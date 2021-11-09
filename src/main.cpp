
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
int main() {

    mnist_loader train("dataset/train-images-idx3-ubyte",
                       "dataset/train-labels-idx1-ubyte", 100);
    mnist_loader test("dataset/t10k-images-idx3-ubyte",
                      "dataset/t10k-labels-idx1-ubyte", 100);
    //std::vector<float> image = train.images(0);

    //std::array<std::array<float, 3>,3> x = {{{0.5,0.5,0.5}, {0.5,0.5,0.5}, {0.5,0.5,0.5}}};
    std::vector<int> dim = {25, 25};
    //use numbers between 0 and 1 or bad things happen
    Model model(0.05, dim);

    for(int i = 0; i<10; ++i){

        for(int j = 0; j < 775; j += 25){
            model.decorrelated_hebbian_learning_openCL(train.images(i).data()+j, 25);
        }
    }




    for(int i = 0; i< 25; i++){
        std::cout<< model.getWeights()[i] << std::endl;
    }

    return 0;
}
//

