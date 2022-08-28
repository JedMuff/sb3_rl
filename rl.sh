#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=16
#SBATCH --output=rl0_%a.out
#SBATCH --array=1-4

module load miniconda
source activate pg
module load mesa/21.2.3-opengl-osmesa-python3-llvm
module load mujoco/2.1.0

if (($SLURM_ARRAY_TASK_ID == 1)) ; then
    srun python rl_multicore.py --model PPO_PandaReach-trpen1.5-v0 --algorithm PPO --ts 8000000 --env PandaReach-trpen1.5-v0 --cpus 16
elif (($SLURM_ARRAY_TASK_ID == 2)) ; then
    srun python rl_multicore.py --model PPO_PandaReach-trpen1.7-v0 --algorithm PPO --ts 8000000 --env PandaReach-trpen1.7-v0 --cpus 16
elif (($SLURM_ARRAY_TASK_ID == 3)) ; then
    srun python rl_multicore.py --model PPO_PandaReach-trpen1.8-v0 --algorithm PPO --ts 8000000 --env PandaReach-trpen1.8-v0 --cpus 16
elif (($SLURM_ARRAY_TASK_ID == 4)) ; then
    srun python rl_multicore.py --model PPO_PandaReach-trpen1.9-v0 --algorithm PPO --ts 8000000 --env PandaReach-trpen1.9-v0 --cpus 16
fi
seff $SLURM_JOB_ID
