

#include "ClHelper.h"
#include "Model.h"
#include <iostream>
#ifdef __APPLE__ //Mac OSX has a different name for the header file
#include <OpenCL/opencl.hpp>
#include "MNIST/mnist_loader.h"
#include <fstream>


#else
#include <CL/opencl.hpp>
#endif

#include "MNIST/mnist_loader.h"
#include <fstream>
#include <chrono>

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

    std::vector<int> dim = {25, 25};
    //use numbers between 0 and 1 or bad things happen
    Model model(0.1, dim);


    int num_segments = 10000;
    std::vector<float> segments;
    for (int i = 0; i < num_segments; ++i) {
        auto segment = train.image_segment();
        segments.insert(segments.end(), segment.begin(), segment.end());
    }


    for (int i = 0; i < 100; ++i) {
        model.dhl_full_gpu(segments.data(), 25, 100, 2);
        printweights(model);
    }




/*

    for(int i = 0; i<num_segments; ++i){
        model.decorrelated_hebbian_learning_CPU(train.image_segment().data(), 25);
    }
*/

    return 0;
}