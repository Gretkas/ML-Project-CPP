void kernel ojasRule(global float* x, global float* w, const float y, const float learning_rate) {
    size_t i = get_global_id(0);
    float temp = x[i]- y*w[i];
    w[i] = w[i] + learning_rate*y*temp;
};

void kernel dhl_y_helper_calc_exponent_vector(global float* x, global float* w, global float* out, const int len){
        size_t gid = get_global_id(0);
        size_t lid = get_local_id(0);
        size_t offset = get_global_offset(0);
        out[gid%len] -= pow(x[offset + lid] - w[gid],2);
}

void kernel dhl_y(global float* exponents, global float* y_dot){
        size_t gid = get_global_id(0);
        float temp = exponents[gid]/0.5;
        exponents[gid] = exp(temp);
        y_dot[0] += exponents[gid];
        barrier(CLK_GLOBAL_MEM_FENCE);
        //calculate all y values
        exponents[gid] = exponents[gid] / y_dot[0];

};


void kernel decorrelatedHebbianLearning(
    global float* x,
    global float* w,
    global float* y,
    const int y_len,
    const int x_len) {

    size_t gid = get_global_id(0);
    float temp =  (x[gid%x_len] - w[gid]) * y[gid%y_len];
    w[gid] += temp;

};


