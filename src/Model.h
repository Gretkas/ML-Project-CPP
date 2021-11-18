//
// Created by Sergio Martinez on 31/10/2021.
//
#include <vector>
#ifndef ML_PROJECT_CPP_MODEL_H
#define ML_PROJECT_CPP_MODEL_H

//weights are stored in one dimension and use dim sizes for tensor operations
//implemented this way in order to ensure contiguous memory, speed should be same
class Model {
public:
    Model(float learning_rate, std::vector<int> &dim_sizes);

    virtual ~Model();

    const float _learning_rate;
    const std::vector<int> _dim_sizes;
    float* conv_weights;
    float* pool_weights;
    float* initial_pool_weights;

    void ojas_rule_openCL(float* x, int length);
    void decorrelated_hebbian_learning_openCL(float* x, int length);

    [[nodiscard]] const float getLearningRate() const;

    [[nodiscard]] const std::vector<int> &getDimSizes() const;

    [[nodiscard]] float *getConvWeights() const;

    [[nodiscard]] float *getPoolWeights() const;


private:
    float ojas_y(const float* x, int length);
    float* dhl_y(const float* x, int length);
    float dhl_y_helper_quotient(float* exponents);
    float* dhl_y_helper_exponent_vector(const float* x, int length);
    float* dhl_y_dot(float* y);
    std::vector<int> find_active(int n);
};



#endif //ML_PROJECT_CPP_MODEL_H
