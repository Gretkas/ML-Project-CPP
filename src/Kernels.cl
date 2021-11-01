void kernel squareArray(global int* input, global int* output) {
   size_t gid = get_global_id(0);
   output[gid] = input[gid] * input[gid];
};

void kernel ojasRule(global float* x, global float* w, const float y, const float learning_rate) {
    size_t i = get_global_id(0);
    float temp = x[i]- y*w[i];
    w[i] = w[i] + learning_rate*y*temp;
};


