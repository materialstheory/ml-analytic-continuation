# Machine-learning approach for the DMFT analytic continuation

[![DOI](https://zenodo.org/badge/485062333.svg)](https://zenodo.org/badge/latestdoi/485062333)

## Overview

This repository contains the code and data for performing the quantum many-body
analytic continuation with a machine learning approach.
It uses a multi-level residual network to continue the DMFT Green's function to
the spectral function as presented in the accompanying paper.

The main functionality of predicting spectral functions with their uncertainty
can be easily run in the demo notebook
`model/AnalyticContinuation.ipynb`: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/materialstheory/ml-analytic-continuation/blob/main/model/AnalyticContinuation.ipynb)

Written by Rong Zhang
under the supervision of Maximilian E. Merkel and Claude Ederer
from the Materials Theory Group at ETH Zurich.

## Content of the repository

* `kernel/`: generates the matrix for the forward transform, eqn (1) from the paper
* `data/`: artificially generated spectral functions, used as the training set.
* `test-data/`: selection of spectral functions or imaginary axis Green’s functions
for demonstration
* `model/`: pre-trained neural network models and the script to train a model on
a different data set. Training requires the content of `kernel/` and `data/`,
whereas the pre-trained model can be run directly on the data from `test-data/`

See also the READMEs inside the folders.

## Copyright and license

Copyright (C) 2022 ETH Zurich, Rong Zhang, Maximilian E. Merkel; Materials Theory Group, D-MATL

This file is part of the repository ml-analytic-continuation.

ml-analytic-continuation is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

ml-analytic-continuation is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with ml-analytic-continuation. If not, see <https://www.gnu.org/licenses/>.
