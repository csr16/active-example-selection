#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

datasets=(agnews sst-2 trec amazon)
seeds=(0 1 2 3 4)  

mkdir -p logs
#####################
for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    temp_yaml="rl_configs/temp_${dataset}_${seed}.yaml"
    
    cp rl_configs/random-agent.yaml "${temp_yaml}"

    sed -i "s/^seed: .*/seed: ${seed}/" "${temp_yaml}"
    sed -i "s/^name: .*/name: t=${dataset}_s=${seed}/" "${temp_yaml}"
    
    sed -i "s|^output_dir: .*|output_dir: ./outputs/active/${dataset}_${seed}|" "${temp_yaml}"
    sed -i "s/^\(  \)task: .*/\1task: ${dataset}/" "${temp_yaml}"
    
    mkdir -p "./outputs/active/${dataset}_${seed}"
    
    log_file="logs/log_${dataset}_${seed}.txt"
    python src/rl/main.py "${temp_yaml}" >> "${log_file}" 2>&1
    
    rm "${temp_yaml}"
  done
done
