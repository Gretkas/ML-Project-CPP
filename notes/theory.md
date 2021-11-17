# Notes on relevant theory

## Definitions
* **Metaparams** - trainable params of the model, f.e.
    * weights
    * biases
    * etc
* **Hyperparams** - params we choose to create a good model
    * learning rate value
    * which activation function to use in hidden layer
    * which activation function to use in output layer
    * NN (neural network) architecture
    * optimizer
    * etc etc etc
* ****

## Notation to help understand math formulas

* **&sigma;** - any non linear (activation) function (f.e. Sigmoid, ReLU, Hyperbolic tangent etc etc)
* **𝜌** - activation function
* **&alpha;** - learning rate
*

## Math
* product derivative
* gradient == derivative
*

## Neural Networks
* **CNN**
    * layers:
        * **convolution** = learn kernels (filters) via Oja's to create feature maps
        * **pooling** = reduce dimensionality and increase receptive field size via DHL (first approximation)
        * **linear** = clustering (second DHL approximation). We decide its architecture, hyperparams etc etc
*

## DHL paper break down
* **radial basis functions** - [explanation](https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons)
* **gaussian neuron** - means that activation function in neurons of the layer is [gaussian](https://www.gabormelli.com/RKB/Gaussian_Activation_Function)
* **performs clustering in the input space** - clustering occurs at the input to neuron at a given layer
* **lateral connection** - is an "intra layer"-connection
    * connections between neurons of the same layer
    * helps reduce the depth of the network (TODO there is research backup)
* **"Fur- thermore, if the sum of the outputs of the gaussians is normalized to one"** - softmax
* **Activation function (Figure 2.1)** - activation of each neuron is softmax
    * TODO entropy
* **Cost function (Figure 2.3)** -
    * **probability density of the input patterns** - is derived from approximation via DHL (normal distribution)
    * **y_i** - output of the neuron
    * **y_j** - output of a neuron of the same layer, excluding y_i
    * the proposed cost function implements approximation via normal distribution
    *
*

## Tips

## Misc
* erstatt lin-softmax med log-softmax i pooling layer

## TODOs
1. tilpasse modellen til å implementere en fullstendig CNN arkitektur
2. lage utregning av antall aktive gaussian centers fra pooling laget for å bestemme antall nevroner i output laget
3. fullstendig forståelse av kobling mellom pooling og output laget
4. 