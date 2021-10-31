void kernel squareArray(global int* input, global int* output) {
   size_t gid = get_global_id(0);
   output[gid] = input[gid] * input[gid];
};

void kernel ojasRule(global float* x, global float* w, global float* y, global float* learn_rate) {
    size_t i = get_global_id(0);
    w[i] += *learn_rate**y*(x[i] - (*y*w[i]));
};


