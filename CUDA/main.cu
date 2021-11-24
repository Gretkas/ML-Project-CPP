#include "helper.cu"
#include "ojas.cu"
#include <iostream>
#include <vector>

/// vente med oppdatering og teste hva forkjellen blir, kan lage to forskjellige metoder og sende inn samme datasett

//må free w og x;
int main() {

    const int num_seg = 10000;  //ant segmenter
    const int len = 25;         //lengden på et segment
    float *w = generate_w(len); // skal bare være 25 lang

    std::cout << "w(0):" << std::endl;
    std::cout << w << std::endl
              << std::endl;

    std::vector<float> x = load_data(num_seg);

    run_ojas(w, x, num_seg, len, true);

    std::cout << "w(" << num_seg << "):" << std::endl;
    std::cout << w << std::endl;

    //Får følgende feil men å ha disse:
    //free(): double free detected in tcache 2
    //delete x;
    //delete w;
}
