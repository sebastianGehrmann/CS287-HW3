#!/bin/bash
#SBATCH --job-name=gpu_win2
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=2000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mail-type=END
#BATCH --mail-user=gehrmann@seas.harvard.edu
#SBATCH -t 0-06:05

th HW3.lua -gpuid 0 -datafile PTB-win-2.hdf5 -lm nn -savePreds true
