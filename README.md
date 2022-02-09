# A Group-Equivariant Autoencoder for Identifying Spontaneously Broken Symmetries in the Ising Model
Devanshu Agrawal, Adrian Del Maestro, Steven Johnston, and James Ostrowski.

### Abstract

We introduce the group-equivariant autoencoder (GE-autoencoder)-- a novel deep neural network method that locates phase boundaries in the Ising universality class by determining which symmetries of the Hamiltonian are broken at each temperature. 
The encoder network of the GE-autoencoder models the order parameter observable associated to a structural phase transition. 
The parameters of the GE-autoencoder are constrained such that the encoder is invariant to the subgroup of symmetries that never break; 
this results in a dramatic reduction in the number of free parameters such that the GE-autoencoder size is independent of the system size. 
The loss function of the GE-autoencoder includes regularization terms that enforce equivariance to the remaining quotient group of symmetries. 
We test the GE-autoencoder method on the 2D classical ferromagnetic and antiferromagnetic Ising models, finding that 
the GE-autoencoder (1) accurately determines which symmetries are broken at each temperature, and (2) estimates the critical temperature with greater accuracy and time-efficiency than a symmetry-agnostic autoencoder, once finite-size scaling analysis is taken into account.


### Description

This repository includes links, code, and scripts to generate the figures in a paper.


### Requirements

- python >= 3.8.10
- pytorch >= 1.9.0
- cvxopt >= 1.2.7
- matplotlib
- numpy
- pandas
- scikit-learn
- scipy


# Data generation

This code expects a subdirectory named `data` containing data generated from an Ising model simulation. 
To generate the data, first make a clone of our [Ising model repository](https://github.com/dagrawa2/ising) in the same parent directory containing this current repository, 
and follow its installation instructions. 
Then to begin the simulation, run the following inside this current repository (make take days to finish):

    python generate_data.py

This will populate the `data` subdirectory.


# Running models and reproducing results

To train all models, run the following (make take days to finish):

    bash train.sh

To analyze the results and generate all plots, run the following:

    python process.py
    python plots.py

The plots will appear in the `results/plots` directory.
