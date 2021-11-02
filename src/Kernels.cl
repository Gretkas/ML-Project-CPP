void kernel squareArray(global int* input, global int* output) {
   size_t gid = get_global_id(0);
   output[gid] = input[gid] * input[gid];
};

void kernel ojasRule(global float* x, global float* w, const float y, const float learning_rate) {
    size_t i = get_global_id(0);
    float temp = x[i]- y*w[i];
    w[i] = w[i] + learning_rate*y*temp;
};

void kernel dhl_y_helper_sum_exponent_vector(global float* exponents, local float* localReduction, global float* out){
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t l_size = get_local_size(0);
    localReduction[lid] = exponents[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = l_size >> 1; i>0; i>>=1){
            if(lid < i){
                localReduction[lid] += localReduction[lid+i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

    if(lid == 0){
        out[get_group_id(0)] = localReduction[0];
    }
}

void kernel dhl_y_helper_calc_quotient(global float* exponents, local float* localReduction, global float* out){
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t l_size = get_local_size(0);
    localReduction[lid] = exponents[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = l_size >> 1; i>0; i>>=1){
            if(lid < i){
                localReduction[lid] = exp(localReduction[lid]) + exp(localReduction[lid+i]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

    if(lid == 0){
        out[get_group_id(0)] = localReduction[0];
    }
}

void kernel dhl_y_helper_calc_exponent_vector(global float* x, global float* w, global float* out, const int len, const int sigma){
    size_t gid = get_global_id(0);
    out[gid] = (-pow(fabs(x[gid%len] - w[gid]),2))/sigma;
}
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


