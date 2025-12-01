#!/bin/bash

# Bash script for evaluating the models

# define the config to use
config_name=bridge_compute_metrics_example

# job script
job_script="slurm_scripts/test/bridge_compute_metrics.sh"

# export eval split size
export BRIDGE_SPLIT_SIZE=100

# set the bounds of the range for the evaluation jobs
local_start=0
local_end=100

for start_idx in $(seq $local_start $local_end); do
    # export eval split settings
    # index of the split
    export BRIDGE_SPLIT_IDX=$start_idx

    # print
    echo "Submitting metrics computation request with start index: $start_idx"
    echo "with config: $config_name"
    
    # launch
    sbatch "$job_script" "$config_path"
done