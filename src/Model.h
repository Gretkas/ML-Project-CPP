//
// Created by Sergio Martinez on 31/10/2021.
//
#include <vector>
#ifndef ML_PROJECT_CPP_MODEL_H
#define ML_PROJECT_CPP_MODEL_H


class Model {
public:
    Model(float learning_rate, float* initial_weights, std::vector<int> &dim_sizes);
    Model(float learning_rate, std::vector<int> &dim_sizes);
    const float _learning_rate;
    const std::vector<int> _dim_sizes;
    float* _weights;
    void ojas_rule_openCL(float* x, int length);

    const float getLearningRate() const;

    const std::vector<int> &getDimSizes() const;

    float *getWeights() const;


private:
    float ojas_y(const float* x, int length);
};



#endif //ML_PROJECT_CPP_MODEL_H
