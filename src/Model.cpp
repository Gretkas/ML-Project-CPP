//
// Created by Sergio Martinez on 31/10/2021.
//

#include "Model.h"
#include <algorithm>
#include <utility>


void Model::ojas_rule(float *x) {
    return;
}

Model::Model(float learning_rate,int dims, const int *dim_sizes) {
    _learning_rate = learning_rate;
    int memsize = 0;
    for(int i = 0; i<dims; ++i){
        memsize += *(dim_sizes + i);
    }
    _weights.reserve(memsize);
    std::fill(_weights.begin(), _weights.end(), 1);
}

Model::Model(float learning_rate, std::vector<float> initial_weights) {
    _learning_rate = learning_rate;
    _weights = std::move(initial_weights);
}
