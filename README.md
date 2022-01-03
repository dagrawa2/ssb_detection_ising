# ssb_detection_ising

Code to reproduce all results in our preprint *Machine learning spontaneously broken symmetries in the Ising model*.


# Requirements

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
