* spectral_function.py generates mock spectral function datasets
* parameter sets: recp1_param.txt recp2_param.txt
* readme.md: how to generate spectral functions for other beta's or other parameter values

# Generate spectral functions for other beta's or other parameter values

* "recp1" and "recp2" denotes recipe 1 and 2 

* generate recp1 and recp2, training and validation datasets, each of size 32000

  ``` 
  python spectral_function.py 32000
  ```
  which generates "A_omega_train.npy" ""A_omega_val.npy" (for recp1) "A_recp2_omega_train.npy" "A_recp2_omega_val.npy"

* change omega grid: line 16, 17 spectral_function.py

* change parameters: *_param.txt