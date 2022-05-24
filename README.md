# Machine-learning approach for the DMFT analytic continuation

## Overview

This repository contains the code and data for performing the quantum many-body
analytic continuation with a machine learning approach.
It uses a multi-level residual network to continue the DMFT Green's function to
the spectral function as presented in the accompanying paper.

The main functionality of predicting spectral functions with their uncertainty
can be easily run in the demo notebook
`model/AnalyticContinuation.ipynb`: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/materialstheory/ml-analytic-continuation/blob/main/model/AnalyticContinuation.ipynb)

Copyright (C) 2022, Materials Theory Group, ETH Zurich, Switzerland

Written by Rong Zhang
under the supervision of Maximilian E. Merkel and Claude Ederer

## Content of the repository

* `kernel/`: generates the matrix for the forward transform, eqn (1) from the paper
* `data/`: artificially generated spectral functions, used as the training set.
* `test-data/`: selection of spectral functions or imaginary axis Green’s functions
for demonstration
* `model/`: pre-trained neural network models and the script to train a model on
a different data set. Training requires the content of `kernel/` and `data/`,
whereas the pre-trained model can be run directly on the data from `test-data/`

See also the READMEs inside the folders.
