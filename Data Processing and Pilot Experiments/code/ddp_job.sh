#!/bin/bash
#SBATCH -N 4
#SBATCH -J lm-inf-ddp
#SBATCH -o logs/ddp-%j.out
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --time=4:00:00

module load gnu/11.2.0
module load openmpi/gnu/4.1.4-gpu
module load cuda

# Setup DDP environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

echo "MASTER_ADDR: $MASTER_ADDR"
echo "WORLD_SIZE: $WORLD_SIZE"

# Bind the required directories
dirname="chef-doeuvre"
mkdir -p /tmpdir/$USER/$dirname
cd /tmpdir/$USER/$dirname

# Run with srun
# We need to pass the SLURM variables to the container and map them to what torch expects
srun apptainer exec --env "PYTHONUSERBASE=${MYENVS}/venv" \
    --env "MASTER_ADDR=$MASTER_ADDR" \
    --env "MASTER_PORT=$MASTER_PORT" \
    --env "WORLD_SIZE=$WORLD_SIZE" \
    --env "RANK=\$SLURM_PROCID" \
    --env "LOCAL_RANK=\$SLURM_LOCALID" \
    --bind /tmpdir,/work,/users \
    --nv /work/conteneurs/sessions-interactives/triton-llvm-3.3.0-calmip-si-latest.sif \
    bash -c "export RANK=\$SLURM_PROCID; export LOCAL_RANK=\$SLURM_LOCALID; python3 /users/m25211/m25211brmn/chef-doeuvre/src/main.py"

echo finished.

