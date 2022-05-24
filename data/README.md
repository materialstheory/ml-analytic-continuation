# Files in the folder

* `spectral_function.py`: generates mock spectral function datasets
* `recp1_param.txt`, `recp2_param.txt`: parameter sets, see Table I of the paper

# Generate spectral functions for other temperatures or other parameter values

* "recp1" and "recp2" denote recipe 1 and 2
* generate recp1 and recp2, training and validation datasets, each of size 32000*9 (a factor of 9 because we extend the dataset according to rules described in our paper)
with the command
  ```
  python spectral_function.py 32000
  ```
  which generates `A_omega_train.npy` and `A_omega_val.npy` for recp1
  or `A_recp2_omega_train.npy` and `A_recp2_omega_val.npy` for recp2
* change omega grid: line 16, 17 `spectral_function.py`
* change parameters: `*_param.txt`
