# Files in the folder

* `train.py`: python script for training
* `cp_*`: trained model-parameter files (checkpoints) for various noise levels
and various data sets. The checkpoints (cp) (saved neural network parameters) of
trained models are named according to the recipe (recp1 or 2), the preprocessing
of the data (5 copies of spectral functions with small fluctuations), the neural
network architecture, number of nodes per layer and noise level used during
training, e.g. “cp_recp2_mult5-res2222_100-n4”
* `AnalyticContinuation.ipynb`: demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/materialstheory/ml-analytic-continuation/blob/main/model/AnalyticContinuation.ipynb)

# How to train models for other kernel matrices

* Go to `kernel/`, generate a kernel matrices
* Go to `data/`, generate new datasets if needed
* In `train.py` in line 14, load the new kernel matrices. Line 20, update omega range
* run the training script

```
python3 train.py "A_omega_train.npy" "A_omega_val.npy" 3 "recp1" 16000
```

* TensorFlow/2.4.0 is used for training, although we believe that other versions probably work
