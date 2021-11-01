//
// Created by Sergio Martinez on 31/10/2021.
//
#include <vector>
#ifndef ML_PROJECT_CPP_MODEL_H
#define ML_PROJECT_CPP_MODEL_H


class Model {
public:
    Model(float learning_rate, std::vector<float> initial_weights);
    Model(float learning_rate, std::vector<float> initial_weights, std::vector<int> &dim_sizes);
    Model(float learning_rate, std::vector<int> &dim_sizes);
    const float _learning_rate;
    const std::vector<int> _dim_sizes;
    std::vector<float> _weights;
    void ojas_rule_openCL(std::vector<float> x);
    [[nodiscard]] const std::vector<float> &getWeights() const;


private:
    float ojas_y(std::vector<float> x);
};



#endif //ML_PROJECT_CPP_MODEL_H
