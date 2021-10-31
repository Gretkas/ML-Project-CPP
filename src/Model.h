//
// Created by Sergio Martinez on 31/10/2021.
//
#include <vector>
#ifndef ML_PROJECT_CPP_MODEL_H
#define ML_PROJECT_CPP_MODEL_H


class Model {
public:
    Model(float learning_rate, std::vector<float> initial_weights);
    Model(float learning_rate,int dims, const int dim[]);
    const float _learning_rate;
    void ojas_rule_openCL(std::vector<float> x);
    std::vector<float> _weights;

    const std::vector<float> &getWeights() const;

private:
    float ojas_y(std::vector<float> x);
};



#endif //ML_PROJECT_CPP_MODEL_H
