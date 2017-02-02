#!/bin/bash -l
#SBATCH --partition=dphys_compute
#SBATCH --time=24:00:00
#SBATCH --job-name="clump"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --array 0-320
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simon.birrer@phys.ethz.ch
#SBATCH --output=/users/sibirrer/Logs/%A.%a.o
#SBATCH --error=/users/sibirrer/Logs/%A.%a.e
#======START===============================
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Running on $SLURM_JOB_CPUS_PER_NODE cpus per node."
echo "Current working directory is `pwd`"

cd $HOME
path=$1
length=320

module load python/2.7.6-gcc-4.8.1

index=${SLURM_ARRAY_TASK_ID}

cd /users/sibirrer/Lenstronomy/lenstronomy/Sensitivity/
python monch_array_script.py $path $index $length

echo "Ending at `date`"
#======END=================================
