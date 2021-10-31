__kernel void squareArray(__global int* input, __global int* output) {
    int i = get_global_id(0);
    output[i] = input[i] * input[i];

}