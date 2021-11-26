#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable //sadly doesnt work for floats
void kernel ojasRule(global float* x, global float* w, const float y, const float learning_rate) {
    size_t i = get_global_id(0);
    float temp = x[i]- y*w[i];
    w[i] = w[i] + learning_rate*y*temp;
};

void kernel dhl_y_helper_calc_exponent_vector(global float* x, global float* w, global float* out){
        size_t offset = get_global_offset(0);
        size_t gid = get_global_id(0) - offset;
        size_t lid = get_local_id(0);
        out[gid] = pow(x[offset + lid] - w[gid],2);


}

//if input is not of size 2^n we need to make it 2^n so it can be reduced.
//many ways to do this, went for a simple one so i dont spend too much time here
void kernel sum_helper(global float* data, const int offset, const int len, global float* out, const int surplus){
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t group_id = get_group_id(0);
    size_t local_size = get_local_size(0);
    if(lid < surplus){
        out[group_id*local_size + lid] = data[group_id*len + lid] + data[group_id*len+lid+offset];
    }
    else{
        out[group_id*local_size + lid] = data[group_id*len + lid];
    }

}

void kernel sum_helper_with_output(global float* data,global float* out, const int offset, const int surplus){
    size_t gid = get_global_id(0);
    if(gid < surplus){
        out[gid] = data[gid] + data[gid+offset];
    }
    else{
        out[gid] = data[gid];
    }
}


// inspired by : https://dournac.org/info/gpu_sum_reduction
__kernel void sum_reduction(__global float* data,__local float* partial_sums, __global float* output)
{
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    //local memory is faster than global
    partial_sums[lid] = data[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = group_size/2; i>0; i >>= 1) {
        if(lid < i) {
            partial_sums[lid] += partial_sums[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0) {
        output[get_group_id(0)] = partial_sums[0];

    }
}

void kernel dhl_y_helper_fraction(global float* exponents, const float sigma){
        size_t gid = get_global_id(0);
        float temp = -exponents[gid]/sigma;
        exponents[gid] = exp(temp);

};

void kernel dhl_y(global float* y, global float* divisor){
        size_t gid = get_global_id(0);

        y[gid] = y[gid]/divisor[0];



};

void kernel sum_helper_with_power_and_output(global float* data,global float* out, const int offset, const int surplus){
    size_t gid = get_global_id(0);
    if(gid < surplus){
        out[gid] = pow(data[gid],2) + pow(data[gid+offset],2);
    }
    else{
        out[gid] = pow(data[gid],2);
    }


}



void kernel decorrelatedHebbianLearning(global float* x, global float* w, global float* y, global float* y_sum, const float learn_rate) {
    size_t offset = get_global_offset(0);
    size_t gid = get_global_id(0) - offset;
    size_t lid = get_local_id(0);
    size_t local_size = get_local_size(0);
    size_t i = gid/local_size;

    //printf("gid : %u,  %f, %f\n",gid, y[i], y_sum[0]);
    w[gid] = w[gid] + (learn_rate * y[i] * (x[offset + lid] - w[gid]) * (y[i] - y_sum[0]));

    //printf("gid : %u,  %f, %u\n",gid, w[gid], 42);

};
/*

 */

void kernel reset_y(global float* y){
    size_t gid = get_global_id(0);
    y[gid] = 0;
}


