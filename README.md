[![Paper](https://img.shields.io/badge/paper-arXiv%3A2202.06319-B31B1B.svg)](https://arxiv.org/abs/2202.06319)
[![DOI](https://zenodo.org/badge/444186389.svg)](https://zenodo.org/badge/latestdoi/444186389)

# A Group-Equivariant Autoencoder for Identifying Spontaneously Broken Symmetries
Devanshu Agrawal, Adrian Del Maestro, Steven Johnston, and James Ostrowski
[arXiv:2202.06319](https://arxiv.org/abs/2202.06319)

### Abstract

We introduce the group-equivariant autoencoder (GE-autoencoder) -- a deep neural network (DNN) method that locates phase boundaries by determining which symmetries of the Hamiltonian have spontaneously broken at each temperature. 
We use group theory to deduce which symmetries of the system remain intact in all phases, and then use this information to constrain the parameters of the GE-autoencoder such that the encoder learns an order parameter invariant to these never-broken symmetries. 
This procedure produces a dramatic reduction in the number of free parameters such that the GE-autoencoder size is independent of the system size. 
We include symmetry regularization terms in the loss function of the GE-autoencoder so that the learned order parameter is also equivariant to the remaining symmetries of the system. 
By examining the group representation by which the learned order parameter transforms, we are then able to extract information about the associated spontaneous symmetry breaking. 
We test the GE-autoencoder on the 2D classical ferromagnetic and antiferromagnetic Ising models, finding that the GE-autoencoder 
(1) accurately determines which symmetries have spontaneously broken at each temperature; 
(2) estimates the critical temperature in the thermodynamic limit with greater accuracy, robustness, and time-efficiency than a symmetry-agnostic baseline autoencoder; and 
(3) detects the presence of an external symmetry-breaking magnetic field with greater sensitivity than the baseline method. 
Finally, we describe various key implementation details, including a new method for extracting the critical temperature estimate from trained autoencoders and calculations of the DNN initialization and learning rate settings required for fair model comparisons.


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

For running `never_broken.py' and reproducing Table III in the paper, the [Gappy](https://github.com/embray/gappy) package must also be installed.


# Data generation

This code expects a subdirectory named `data` containing data generated from an Ising model simulation. 
To generate the data, first make a clone of our [Ising model repository](https://github.com/dagrawa2/ising) in the same parent directory containing this current repository, 
and follow its installation instructions. 
Then to begin the simulation, run the following inside this current repository (make take days to finish):

    python generate_data.py

This will populate the `data` subdirectory. 
To aggregate the generated data across temperatures, run the following:

    python aggregate_data.py


# Running models and reproducing results

## For scalar order parameters

To train all models, run the following for every integer value of `[fold]' between 1 and 8 (make take days to finish):

    bash train.sh [fold]
    python anomaly_detection.py

To analyze the results and generate all plots, run the following:

    python process.py
    python plots.py

The plots will appear in the `results/plots` directory.

To additionally reproduce the quantification of equivariance of the baseline-autoencoder described in Sec. IV.B of the paper, run the following:

    bash baseline_equivariance.sh
    python process_baseline_equivariance.py


## For vector order parameters

To aggregate the necessary data, run the following:

    python aggregate_data_2D.py

To train all models, analyze the results, and reproduce Table II in the paper, run the following for every integer value of `[fold]' between 1 and 8 (make take days to finish):

    bash train_2D.sh [fold]
    python process_2D.py

Finally, to reproduce Table III in the paper, run the following:

    python never_broken.py
