#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=16
#SBATCH --output=rl.out

module load miniconda
source activate pg
module load mesa/21.2.3-opengl-osmesa-python3-llvm
module load mujoco/2.1.0

#srun python rl_multicore.py --model PPO_RandomPegBox-v0-2 --algorithm PPO --ts 8000000 --env RandomPegBox-v0 --cpus 16
#srun python rl_multicore.py --model PPO_PandaReach-PosCtrl-v0 --algorithm PPO --ts 8000000 --env PandaReach-PosCtrl-v0 --cpus 16
#srun python rl_multicore.py --model PPO_RandomPegCylinder-v0 --algorithm PPO --ts 8000000 --env RandomPegCylinder-v0 --cpus 16
srun python rl_multicore.py --model PPO_PandaHockey-v0 --algorithm PPO --ts 8000000 --env PandaHockey-v0 --cpus 16
srun python rl_multicore.py --model PPO_PandaHockey-Dpos-v0 --algorithm PPO --ts 8000000 --env PandaHockey-Dpos-v0 --cpus 16
srun python rl_multicore.py --model PPO_PandaHockey-Dvel-v0 --algorithm PPO --ts 8000000 --env PandaHockey-Dvel-v0 --cpus 16

# srun python rl_multicore.py --model PPO_PandaBall-Random-PosCtrl-v0 --algorithm PPO --ts 8000000 --env PandaBall-Random-PosCtrl-v0 --cpus 16
# srun python rl_multicore.py --model PPO_PandaBall-v0 --algorithm PPO --ts 8000000 --env PandaBall-v0 --cpus 16
# srun python rl_multicore.py --model PPO_PandaBasketball-Dvel-v0 --algorithm PPO --ts 8000000 --env PandaBasketball-Dvel-v0 --cpus 16
# srun python rl_multicore.py --model PPO_PandaBasketball-v0 --algorithm PPO --ts 8000000 --env PandaBasketball-v0 --cpus 16
# srun python rl_multicore.py --model PPO_PandaFindBox-v0 --algorithm PPO --ts 8000000 --env PandaFindBox-v0 --cpus 16
# srun python rl_multicore.py --model PPO_PandaReach-v0 --algorithm PPO --ts 8000000 --env PandaReach-v0 --cpus 16
# srun python rl_multicore.py --model PPO_RandomPandaSwingPeg-v0 --algorithm PPO --ts 8000000 --env RandomPandaSwingPeg-v0 --cpus 16

#PandaBallNoreward-v0
#PandaBallNoreward-v1

#PandaBasketball-Dpos-200ms-v0
#PandaBasketball-Dpos-100ms-v0
#PandaBasketball-Dpos-v0
#PandaBasketball-Dpos-NoCtrlPen-v0
#PandaBasketballBouncy-v0

#PandaReach-ImpCtrl-v0
#PandaReach-PosCtrl-GoalA-v0
#PandaReach-PosCtrl-GoalB-v0
#PandaReach-PosCtrl-GoalC-v0