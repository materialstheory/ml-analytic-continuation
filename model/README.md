# How to train models for other kernel matrices

* Go to kernel/, generate a kernel matrices

* Go to data/, generate new datasets if needed

* In model_multiple_srun.py line 14, load the new kernel matrices. Line 20, update omega range

  

* run the script

```
python  models_multiple_srun.py "A_omega_train.npy" "A_omega_val.npy" 3 "Arsenault" 16000
```

