#!/bin/bash -l

#SBATCH --partition=dphys_compute
#SBATCH --time=24:00:00
#SBATCH --job-name="map_sensitivity"
#SBATCH --ntasks=1
#SBATCH --mem=2000
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simon.birrer@phys.ethz.ch
#SBATCH --output=/users/sibirrer/Logs/test.%j.o
#SBATCH --error=/users/sibirrer/Logs/test.%j.e
#======START===============================
cd $HOME
path=$1
module load python/2.7.6-gcc-4.8.1

cd /users/sibirrer/Lenstronomy/lenstronomy/Sensitivity/
python compute_map.py $path
#======END=================================