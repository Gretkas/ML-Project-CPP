#include "helper.cu"
#include "ojas.cu"
#include <iostream>
#include <vector>

/// vente med oppdatering og teste hva forkjellen blir, kan lage to forskjellige metoder og sende inn samme datasett

//må free w og x;
int main() {
    const int num_neurons = 100;                    //ant nevroner som trenes
    const int num_seg = 100;                        //ant segmenter/bilder
    const int len = (28*28);                        //lengden på et segment/bilde Patch:(5*5), Bilde:(28*28)
    float *w = generate_w(len * num_neurons);       //skal bare være 25 lang * ant nevroner

    std::cout << "w(0):" << std::endl;
    std::cout << w << std::endl
              << std::endl;

    std::vector<float> x = load_data(num_seg, false); //Set lik false dersom su ønsker å bruke bilder istedenfor bildepatcher/segmenter

    run_ojas(w, x, num_seg, len, num_neurons, false);

    std::cout << "w(" << num_seg << "):" << std::endl;
    std::cout << w << std::endl;

    //Får følgende feil men å ha disse:
    //free(): double free detected in tcache 2
    //delete x;
    //delete w;
}
