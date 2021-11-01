void kernel squareArray(global int* input, global int* output) {
   size_t gid = get_global_id(0);
   output[gid] = input[gid] * input[gid];
};

void kernel ojasRule(global float* x, global float* w, const float y, const float learning_rate) {
    size_t i = get_global_id(0);
    float temp = x[i]- y*w[i];
    w[i] = w[i] + learning_rate*y*temp;
};

void kernel dhl_y_helper_quotient(global const float*, local float* localReduction const int sigma, float* out){
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t l_size = get_local_size(0);

}

void kernel dhl_y_helper_exponent(global const float*, local float* localReduction const int sigma, float* out){
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t l_size = get_local_size(0);

}
void kernel decorrelatedHebbianLearning(
    global float* x,
    global float* w,
    global float* y,
    const float y_dot,
    const float learning_rate,
    const int x_len) {
    size_t gid = get_global_id(0);
    float y_1 = learning_rate * y[gid] * (y[gid] - y_dot);
    for(int i = 0; i < x_len; ++i){
        float temp =  (x[i] - w[x_len*gid +i]) * y_1;
        w[x_len*gid+i] += temp;
    }
};


