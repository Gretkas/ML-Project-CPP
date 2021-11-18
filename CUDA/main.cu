#include "../src/MNIST/mnist_loader.cpp"
#include <iostream>
#include <random>
#include <typeinfo>

using namespace std;

__global__ void ojas_rule(float *x, float *w, const float y, const float learning_rate) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = x[i] - y * w[i];
    w[i] = w[i] + learning_rate * y * temp;
};

//Burde denne kjøres på divice eller på
__device__ float y(const float *x, const float *w, int length) {
    float y = 0;
    for (int i = 0; i < length; ++i) {
        y += w[i] * x[i];
    }
    return y;
}

__host__ void run_ojas(float *w, const float *x) {

    const int n = 0; //length of arrays
    const float y = 0;
    const float learning_rate = 0;
    float *c_w, *c_x;

    //Initilaisere verdier mellom 0 og 1

    cudaMalloc(&c_w, sizeof(w)); // er sizeof her riktig?? Ja er vel det
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
}

__host__ float *load_data() {
    mnist_loader train("datasets/train-images.idx3-ubyte",
                       "datasets/train-labels.idx1-ubyte", 100);
    mnist_loader test("datasets/t10k-images.idx3-ubyte",
                      "datasets/t10k-labels.idx1-ubyte", 100);

    std::vector<float> img_seg = train.image_segment();

    float *x;
    x = img_seg.data();
    return x;
}

//Husk å free arrayet etter bruk!!
__host__ float *generate_w(const int len) {
    float *w;
    w = new float[len];
    srand((unsigned)time(0));

    for (int i = 0; i < len; ++i) {
        w[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (1 + 1))) - 1.0;
    }
    return w;
}

//må free w og x;
int main() {

    int len = 25; //lengden på diverse arrays
    float *w = generate_w(len);

    float *x = load_data();

    for (int i = 0; i < len; ++i) {
        cout << x[i] << endl;
    }
    //run_ojas(w, x);

    //Får følgende feil men å ha disse:
    //free(): double free detected in tcache 2
    //delete x;
    //delete w;
}
