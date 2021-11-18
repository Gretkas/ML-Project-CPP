
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
                out += std::to_string(m.getPoolWeights()[25*j+i]);
                std::cout << out << std::endl;
                out = "";
            }
            else{
                out += std::to_string(m.getPoolWeights()[25*j+i]) + " ";
            }
        }
    }

}

void printConvweights(const Model& m){
    int size = m.getDimSizes()[1];
    std::string out;
    for(int i = 0; i < size; ++i){
        if(i%25 == 24){
            out += std::to_string(m.getConvWeights()[i]);
            std::cout << out << std::endl;
            out = "";
        }
        else{
            out += std::to_string(m.getConvWeights()[i]) + " ";
        }
    }


}
void printbest(std::vector<int> best, const Model& m){
    for(int i = 0; i < best.size(); ++i){
        std::string out;
        for(int j = 0; j < 25; ++j){
            if(j%25 == 24){
                out += std::to_string(m.getPoolWeights()[25*best[i]+j]);
                std::cout << out << std::endl;
                out = "";
            }
            else{
                out += std::to_string(m.getPoolWeights()[25*best[i]+j]) + " ";
            }
        }

    }
}

void printseg(std::vector<float> seg){
    std::string out;
    for(int i = 1; i < 26; ++i){
        if(i%5 == 0){
            out += std::to_string(seg[i-1]);
            std::cout << out << std::endl;
            out = "";
        }
        else{
            out += std::to_string(seg[i-1]) + " ";
        }
    }
    std::cout << std::endl;

}


int main() {

    mnist_loader train("dataset/train-images-idx3-ubyte",
                       "dataset/train-labels-idx1-ubyte", 10000);
    mnist_loader test("dataset/t10k-images-idx3-ubyte",
                      "dataset/t10k-labels-idx1-ubyte", 10000);
    //std::vector<float> image = train.images(0);

    //std::array<std::array<float, 3>,3> x = {{{0.5,0.5,0.5}, {0.5,0.5,0.5}, {0.5,0.5,0.5}}};
    std::vector<int> dim = {25, 25};
    //use numbers between 0 and 1 or bad things happen
    Model model(0.01, dim);
    //printweights( model);

    for(int i = 0; i<10000; ++i){
        //model.ojas_rule_openCL(train.image_segment().data(), 25);
        model.decorrelated_hebbian_learning_openCL(train.image_segment().data(), 25);
        if(i%100 == 0){

            printweights(model);

        }

    }


    // x_vec -> DHL w = n vec

    return 0;
}