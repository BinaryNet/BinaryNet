# Train-time

## Motivations

The goal of this subrepository is to enable the reproduction of the benchmark results reported in the article 
BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.

## Requirements

* Python, Numpy, Scipy
* [Theano](http://deeplearning.net/software/theano/install.html) (Bleeding edge version)
* [Pylearn2](http://deeplearning.net/software/pylearn2/)
* [Lasagne](http://lasagne.readthedocs.org/en/latest/user/installation.html)
* [PyTables](http://www.pytables.org/usersguide/installation.html) (only for the SVHN dataset)
* a fast Nvidia GPU (or a large amount of patience)

## MNIST MLP

    python mnist.py
    
This python script trains an MLP on MNIST with BinaryNet.
It should run for about 6 hours on a Titan Black GPU.
The final test error should be around **0.96%**.

## CIFAR-10 ConvNet

    python cifar10.py
    
This python script trains a ConvNet on CIFAR-10 with BinaryNet.
It should run for about 23 hours on a Titan Black GPU.
The final test error should be around **11.40%**.

## SVHN ConvNet

    python svhn.py
    
This python script trains a ConvNet on SVHN with BinaryNet.
It should run for about 2 days on a Titan Black GPU.
The final test error should be around **2.80%**.

## How to play with it

The python scripts mnist.py, cifar10.py and svhn.py contain all the relevant hyperparameters.
It is very straightforward to modify them.
binary_net.py contains custom Lasagne layers which support BinaryNet.

Have fun!
