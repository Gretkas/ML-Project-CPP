#include "../src/MNIST/mnist_loader.cpp"
#include <iostream>
#include <random>
#include <typeinfo>

/// vente med oppdatering og teste hva forkjellen blir, kan lage to forskjellige metoder og sende inn samme datasett

using namespace std;

__global__ void w_ojas(float *x, float *w, const float y, const float learning_rate) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = x[i] - y * w[i];
    w[i] = w[i] + learning_rate * y * temp;
};

__device__ float y_ojas(const float *w, const float *x) {
    const int length = 25;
    float y = 0;
    for (int i = 0; i < length; ++i) {
        y += w[i] * x[i];
    }
    return y;
}

__global__ void ojas_rule(float *x, float *w, const float learning_rate, const int num, const int seg_len) {
    float y;

    printf("test\n");

    for (int i = 0; i < num; ++i) {
        //printf("const char *, ...");
        //printf("%f", x[i]);
        //Må sende inn riktig deler av x
        //y = y_ojas(w, x);
        //w_ojas<<<1, 25>>>(x, w, y, learning_rate);
        //cudaDeviceSynchronize();
    }
}

__host__ void run_ojas(float *w, vector<vector<float>> vec_x, const int num, const int seg_len) {

    float x[num][seg_len];
    /*for (int i = 0; i < num; ++i) {
        for (int j = 0; j < seg_len; ++j) {
            x[i][j] = (*vec_x.data())[j];
            printf("%f", (*vec_x.data())[j]);
        };
    }*/
    cout << endl;
    cout << endl;
    int a = 0;
    int b = 0;
    for (auto i : vec_x) {
        for (auto j : i) {
            x[a][b] = j;
            b++;
        }
        a++;
    }

    cout << endl;
    cout << endl;
    cout << endl;

    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < seg_len; ++j) {
            printf("%f", x[i][j]);
        };
    }

    cout << endl;
    cout << endl;

    const float learning_rate = 0.1;
    float *d_w, *d_x;

    cudaMalloc(&d_w, sizeof(w));
    cudaMalloc(&d_x, sizeof(x));

    cudaMemcpy(d_w, w, sizeof(w), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, sizeof(x), cudaMemcpyHostToDevice);

    ojas_rule<<<1, 1>>>(d_x, d_w, learning_rate, num, seg_len);

    cudaDeviceSynchronize();
    cudaMemcpy(w, d_w, sizeof(w), cudaMemcpyDeviceToHost);

    cudaFree(d_w);
    cudaFree(d_x);
}

__host__ vector<vector<float>> load_data(const int num) {
    mnist_loader train("datasets/train-images.idx3-ubyte",
                       "datasets/train-labels.idx1-ubyte", 100);
    mnist_loader test("datasets/t10k-images.idx3-ubyte",
                      "datasets/t10k-labels.idx1-ubyte", 100);

    vector<vector<float>> segments;

    for (int i = 0; i < num; ++i) {
        segments.emplace_back(train.image_segment());
    }

    return segments;
}

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

    const int num_seg = 10;     //ant segmenter
    const int len = 25;         //lengden på et segment
    float *w = generate_w(len); // skal bare være 25 lang

    cout << "w(0):" << endl;
    for (int j = 0; j < len; ++j) {
        cout << w[j] << ", ";
    }
    cout << endl;

    vector<vector<float>> x = load_data(num_seg);

    run_ojas(w, x, num_seg, len);

    cout << "W(1):" << endl;
    for (int i = 0; i < 25; ++i) {
        cout << w[i] << ", ";
    }
    cout << endl;

    //Får følgende feil men å ha disse:
    //free(): double free detected in tcache 2
    //delete x;
    //delete w;
}
