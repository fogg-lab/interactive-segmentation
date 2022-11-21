#!/bin/bash

### Multi-gpu training via distributed training on one machine ###

# Usage:
# 1: Change the values of the variables in section 1 below to match your setup.
# 2: Run the script with one of these commands:
#   ./execute_training_dist.sh   (start all processes)
#   ./execute_training_dist.sh 1 (skip first rank 0 process, assuming it is already running)

initial_rank=$1
if [[ ${initial_rank} != 1 ]]; then initial_rank=0; fi

# Section 1. Set local and environment variables
project_dir="/nfs/hpc/share/wigginno/branching/interactive-segmentation"
script_path="${project_dir}/train.py"
model_path="${project_dir}/models/segformerB3_mix.py"
pretrained_weights="${project_dir}/pretrained/segformer_b3/mit_b3.pth"
dataset_path="${project_dir}/config/mix_datasets_config.yml"
num_gpus=4
workers=16
batch_size=64
exp_name="segformerB3_mix"
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=5000
export NUMEXPR_MAX_THREADS=64

# 2. Start distributed training
# Loop from procnum=0 to procnum=num_gpus-1
for ((procnum=${initial_rank}; procnum<num_gpus; procnum++))
do
    export RANK=${procnum}
    python ${script_path} ${model_path} --pretrained_weights=${pretrained_weights} \
        --dataset_path=${dataset_path} --ngpus=${num_gpus} --workers=${workers} \
        --batch-size=${batch_size} --exp-name=${exp_name} --local_rank=${procnum} \
        > output_${procnum}.log 2>&1 &
done
