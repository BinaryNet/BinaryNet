# Run-time

## Motivations

This subrepository makes available the XNOR (and baseline) GPU kernel(s) described in the article: 
BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.

## Requirements

* Python, Numpy, Scipy
* [Theano](http://deeplearning.net/software/theano/install.html) (Bleeding edge version)
* [Pylearn2](http://deeplearning.net/software/pylearn2/)
* [Lasagne](http://lasagne.readthedocs.org/en/latest/user/installation.html)
* a fast Nvidia GPU

##  Matrix multiplication

    python binary_gemm.py
    
This script performs 4096x4096x4096 matrix-matrix multiplications with our XNOR and baseline GPU kernels.
The two kernels return exactly the same output when their inputs are constrained to -1 or +1 (but not otherwise).
The XNOR kernel is more than an order of magnitude faster than the baseline kernel.

## MNIST MLP Runtime

First, you need to get some trained parameters:

    python ../Train-time/mnist.py    
    
Then, you can run the MNIST MLP using our XNOR GPU kernel:

    python mnist.py
    
The running time will largely depend on the GPU you use.
The test error rate should be around 0.96%.
You can compare these results with the baseline kernel by modifying the 61-62th lines of the script (do not worry, it is very straightforward).