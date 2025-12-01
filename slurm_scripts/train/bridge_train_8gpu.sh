#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --gres=gpu:8                                    ## Number of GPUs per node
#SBATCH --ntasks-per-node=8                             ## Number of tasks per node
#SBATCH --cpus-per-task=8                               ## CPU cores per task
#SBATCH --mem=320G                                      ## Memory per node
#SBATCH --time=60:00:00                                 ## Walltime
#SBATCH --job-name=train_bridge                         ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events, e.g., NONE, BEGIN, END, FAIL, ALL.
#SBATCH --mail-user=<user_email_address>


# change directory (Update)
cd "<full path to c-cubed>/dynuq"

# path to the UV environment directory (Update)
UV_PATH="path to the UV environment directory"

# initialize session
source "${UV_PATH}/envs/dynuq/bin/activate"

# export environment variables
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# number of GPUS
num_gpu=8

# config path
config_path=configs/train_bridge_256_example.yaml

# run the training script with torchrun
# pre-specify a port
torchrun --nproc_per_node=${num_gpu} --master_port 29501 scripts/train.py --config ${config_path}
