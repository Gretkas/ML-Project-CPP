# ML-Project-CPP

## Requirements
OpenCL headers and c++ bindings for your OS. For MacOS you can follow this guide: https://ham.id.au/getting-started-with-opencl-and-cpp-on-macos-catalina/
MNIST dataset: http://yann.lecun.com/exdb/mnist/

## Setup
```
git clone https://github.com/Gretkas/ML-Project-CPP.git
cd ML-Project-CPP
mkdir build
cd build
cmake ..
make
```
after this is done move your MNIST files into a folder call dataset, and move the dataset folder into the build directory.

## Dependencies

Mnist loader: https://github.com/arpaka/mnist-loader

OpenCL cmake file: https://github.com/JanosGit/CrossPlatformOpenCLCMakeLists


## Visualization
See: https://gist.github.com/bubuto355/f3b75902458cdaf4745e3dda67641cc5



