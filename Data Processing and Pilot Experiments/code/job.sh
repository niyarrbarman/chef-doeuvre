#!/bin/bash
#SBATCH -N 1 
#SBATCH -J chef-doeuvre
#SBATCH -o logs/inf-%j.out
#SBATCH -n 1 
#SBATCH --gres=gpu:1 
#SBATCH -p shared 
#SBATCH --ntasks-per-node=1 
#SBATCH --time=4:00:00

module load gnu/11.2.0
module load openmpi/gnu/4.1.4-gpu
module load cuda

# Bind the required directories and run the training script
dirname="chef-doeuvre"
mkdir /tmpdir/$USER/$dirname
cd /tmpdir/$USER/$dirname


apptainer exec --env "PYTHONUSERBASE=${MYENVS}/venv" --bind /tmpdir,/work,/users --nv /work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif \
bash -c "python3 /users/m25211/m25211brmn/chef-doeuvre/src/main.py"
echo finished.