#include "../src/MNIST/mnist_loader.cpp"
#include <iostream>
#include <random>
#include <typeinfo>

/// vente med oppdatering og teste hva forkjellen blir, kan lage to forskjellige metoder og sende inn samme datasett

using namespace std;

__global__ void ojas_rule(float *x, float *w, const float y, const float learning_rate) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = x[i] - y * w[i];
    w[i] = w[i] + learning_rate * y * temp;
};

//Burde denne kjøres på divice eller på
__host__ float y_ojas(const float *w, const float *x, int length) {
    float y = 0;
    cout << "x - y_ojas:" << endl;
    for (int i = 0; i < length; ++i) {
        cout << x[i];
        y += w[i] * x[i];
    }
    cout << endl
         << endl;
    return y;
}

__host__ void run_ojas(float *w, vector<float> x, const int len) {

    const float y = y_ojas(w, x.data(), len);
    cout << "y: " << y << endl;
    const float learning_rate = 0.1;
    float *c_w, *c_x;

    cout << "x - run ojas:" << endl;
    for (int i = 0; i < len; i++) {
        cout << x[i];
    }
    cout << endl
         << endl;

    cudaMalloc(&c_w, sizeof(w)); // er sizeof her riktig?? Ja er vel det
    cudaMalloc(&c_x, sizeof(x));

    cudaMemcpy(c_w, w, sizeof(w), cudaMemcpyHostToDevice);
    cudaMemcpy(c_x, x.data(), sizeof(x), cudaMemcpyHostToDevice);

    //Må fikse her. Skal bare ha 25 threads
    /*
    int block_size = 256;                          //number of threads per block
    int grid_size = (n + block_size) / block_size; //number of blocks
    */

    ojas_rule<<<1, 25>>>(c_x, c_w, y, learning_rate);

    cudaDeviceSynchronize();
    cudaMemcpy(w, c_w, sizeof(w), cudaMemcpyDeviceToHost);

    cout << "W(i+1:)" << endl;
    for (int i = 0; i < len; ++i) {
        cout << w[i] << ", ";
    }
    cout << endl;

    cudaFree(c_w);
    cudaFree(c_x);
}

__host__ vector<float> load_data() {
    mnist_loader train("datasets/train-images.idx3-ubyte",
                       "datasets/train-labels.idx1-ubyte", 100);
    mnist_loader test("datasets/t10k-images.idx3-ubyte",
                      "datasets/t10k-labels.idx1-ubyte", 100);

    std::vector<float> img_seg = train.image_segment();
    cout << "x - img_seg:" << endl;
    for (auto i : img_seg) {
        cout << i;
    }
    cout << endl
         << endl;
    /*
    float x = [img_seg.size()];
    //x = img_seg.data();
    //cout << "x - arr func:" << endl;
    for (size_t i = 0; i < img_seg.size(); ++i) {
        x[i] = img_seg.data()[i];
    }
    */
    return img_seg;
}

//Husk å free arrayet etter bruk!!
__host__ float *generate_w(const int len) {
    float *w;
    w = new float[len];
    srand((unsigned)time(0));

    for (int i = 0; i < len; ++i) {
        w[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (1 + 1)) - 1.0); //initailisere mellom 0 - 1 eller -1 - 1
    }
    return w;
}

//må free w og x;
int main() {

    int len = 25; //lengden på diverse arrays
    float *w = generate_w(len);
    for (int i = 0; i < len; ++i) {
        cout << w[i] << ", ";
    }
    cout << endl;

    vector<float> x = load_data();
    //er rar her?
    cout << "x - main:" << endl;
    for (size_t i = 0; i < len; i++) {
        cout << x[i];
    }
    cout << endl
         << endl;
    run_ojas(w, x, len);

    //Får følgende feil men å ha disse:
    //free(): double free detected in tcache 2
    //delete x;
    //delete w;
}
