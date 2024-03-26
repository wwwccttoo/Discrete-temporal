#!/bin/bash
#$ -pe mpi-24 72
#$ -q long
#$ -N "CASE2_pos128"


module load mpich/3.3/gcc/8.5.0
module load matlab/2023b
eval "$(/YOUR_CONDA_PATH shell.bash hook)"
conda activate thickness
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software01/matlab/R2023b/bin/glnxa64
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mpirun -n 72 python -m mpi4py.run CASE2_TVBOpos_128.py
