#!/bin/bash
declare -a label=('A' 'e')
declare -a trainset=('"A_omega_train.npy"' '"A_easy_omega_train.npy"')
declare -a valset=('"A_omega_val.npy"' '"A_easy_omega_val.npy"')
declare -a longlabel=('"Arsenault"' '"easy"')
envvar='$SLURM_CPUS_PER_TASK'
for j in 0 1
do 
    for i in 2 3 4
    do 
cat>models_multiple_srun_${label[$j]}${i}.sh<<EOF
#!/bin/bash -l
 
#SBATCH --job-name=1e-${i}${label[$j]}
#SBATCH --time=05:00:00
#SBATCH --switches=1
#SBATCH --nodes=1
#SBATCH --output=${label[$j]}1e${i}.out
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --partition=normal
#SBATCH --account=s1128

module load daint-gpu
module load TensorFlow/2.4.0-CrayGNU-21.09

export OMP_NUM_THREADS=${envvar}
#conda activate /apps/daint/UES/6.0.UP04/sandboxes/sarafael/miniconda-tf2.3

srun echo "running..."
srun which python

srun python3  models_multiple_srun.py ${trainset[$j]} ${valset[$j]} ${i} ${longlabel[$j]}
EOF

    done
done

