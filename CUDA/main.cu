#include <iostream>

__global__ void ojas_rule(float *x, float *w, const float y, const float learning_rate) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  float temp = x[i] - y * w[i];
  w[i] = w[i] + learning_rate * y * temp;
};

int main() {
  const int n = 0; //length of arrays
  const float y = 0;
  const float learning_rate = 0;
  float *w, *x;
  float *c_w, *c_x;

  //Initilaisere verdier mellom 0 og 1

  cudaMalloc(&c_w, sizeof(w));
  cudaMalloc(&c_x, sizeof(x));

  cudaMemcpy(c_w, w, sizeof(w), cudaMemcpyHostToDevice);
  cudaMemcpy(c_x, x, sizeof(x), cudaMemcpyHostToDevice);

  int block_size = 256;                          //number of threads per block
  int grid_size = (n + block_size) / block_size; //number of blocks
  ojas_rule<<<grid_size, block_size>>>(c_x, c_w, y, learning_rate);

  cudaDeviceSynchronize();
  cudaMemcpy(w, c_w, sizeof(w), cudaMemcpyDeviceToHost);

  cudaFree(c_w);
  cudaFree(c_x);
  free(w);
  free(x);
}
