# ml-analytic-continuation
This repository contains codes and some data for the machine learning for the analytic continuation project.

kernel

* how to construct the kernel matrix: Aw2Gl.ipynb
* which requires the preblur method from triqs: kernels.py omega_meshes.py preblur.py
* an example beta = 40, [-8, 8], N = 800, kernel matrix: o2l_b40.npy
* readme.md: how to generate other kernel matrices

model

* models_multiple_srun.py
* trained model files for various noise levels and various dataset
* AnalyticContinuation.ipynb for demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/materialstheory/ml-analytic-continuation/blob/main/model/AnalyticContinuation.ipynb)
* readme.md: how to train models for other kernel matrices

data

* Arsenault_spectral_function_v3.py generates mock datasets
* for both parameter sets: Arsenault_param.txt Arsenault_easy_param.txt
* readme.md: generate spectral functions for other beta's or other parameter values

test-data

* SVO: SVO at various beta, QMC and FTPS
* mock Gaussians
