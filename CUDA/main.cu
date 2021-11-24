#include "helper.cu"
#include <iostream>
#include <random>
#include <typeinfo>
#include <vector>

/// vente med oppdatering og teste hva forkjellen blir, kan lage to forskjellige metoder og sende inn samme datasett

using namespace std;

__global__ void w_ojas(float *x, float *w, const float y, const float learning_rate) {
    size_t i = threadIdx.x;
    float temp = x[i] - y * w[i];
    w[i] = w[i] + learning_rate * y * temp;
};

__device__ float y_ojas(const float *w, const float *x, const int len) {
    float y = 0;
    for (int i = 0; i < len; ++i) {
        y += w[i] * x[i];
    }
    return y;
}

__global__ void ojas_rule(float *x, float *w, const float learning_rate, const int num, const int len) {
    float y;

    for (int i = 0; i < num; ++i) {
        float *x_start = &(x[i * len]); //Må sende inn riktig deler av x
        y = y_ojas(w, x_start, len);

        w_ojas<<<1, len>>>(x_start, w, y, learning_rate);
        cudaDeviceSynchronize();
    }
}

__host__ void run_ojas(float *w, vector<float> vec_x, const int num, const int len) {

    float *x = vec_x.data();
    float *d_w, *d_x;
    const float learning_rate = 0.1;
    const size_t x_size = sizeof(*x) * num * len;
    const size_t w_size = sizeof(*w) * len;

    cudaMalloc(&d_w, w_size);
    cudaMalloc(&d_x, x_size);

    cudaMemcpy(d_w, w, w_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, x_size, cudaMemcpyHostToDevice);

    ojas_rule<<<1, 1>>>(d_x, d_w, learning_rate, num, len);

    cudaDeviceSynchronize();
    cudaMemcpy(w, d_w, w_size, cudaMemcpyDeviceToHost);

    cudaFree(d_w);
    cudaFree(d_x);
}

//må free w og x;
int main() {

    const int num_seg = 100000; //ant segmenter
    const int len = 25;         //lengden på et segment
    float *w = generate_w(len); // skal bare være 25 lang

    cout << "w(0):" << endl;
    for (int j = 0; j < len; ++j) {
        cout << w[j] << ", ";
    }
    cout << endl;
    vector<float> x = load_data(num_seg);

    run_ojas(w, x, num_seg, len);

    cout << endl;
    cout << "w(" << num_seg << "):" << endl;
    for (int i = 0; i < 25; ++i) {
        cout << w[i] << ", ";
    }
    cout << endl;

    //Får følgende feil men å ha disse:
    //free(): double free detected in tcache 2
    //delete x;
    //delete w;
}
