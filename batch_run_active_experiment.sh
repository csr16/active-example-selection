#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3

# datasets=(agnews sst-2 trec amazon winowhy epistemic_reasoning hyperbaton timedial aqua)
# seeds=(0 1 2)  

mkdir -p logs
#####################
# for dataset in "${datasets[@]}"; do
#   for seed in "${seeds[@]}"; do
#     temp_yaml="rl_configs/temp_${dataset}_${seed}.yaml"
    
#     cp rl_configs/random-agent.yaml "${temp_yaml}"

#     sed -i "s/^seed: .*/seed: ${seed}/" "${temp_yaml}"
#     sed -i "s/^name: .*/name: t=${dataset}_s=${seed}/" "${temp_yaml}"
    
#     sed -i "s|^output_dir: .*|output_dir: ./outputs/active/${dataset}_${seed}|" "${temp_yaml}"
#     sed -i "s/^\(  \)task: .*/\1task: ${dataset}/" "${temp_yaml}"
    
#     mkdir -p "./outputs/active/${dataset}_${seed}"
    
#     log_file="logs/log_${dataset}_${seed}.txt"
#     python src/rl/main.py "${temp_yaml}" # >> "${log_file}" 2>&1
    
#     rm "${temp_yaml}"
#   done
# done

datasets=(agnews)
seeds=(0)  

for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    temp_yaml="rl_configs/temp_agnews_same_${dataset}_${seed}.yaml"
    cp rl_configs/agnews-same-task.yaml "${temp_yaml}"

    sed -i "s/^seed: .*/seed: ${seed}/" "${temp_yaml}"
    sed -i "s|^\(\s*load_transitions:\).*|\1 ./outputs/active/${dataset}_${seed}/ckpts/transitions_1.ckpt|" "${temp_yaml}"
    sed -i "s/^\(\s*task:\).*/\1 ${dataset}/" "${temp_yaml}"

    sed -i "s|^output_dir: .*|output_dir: ./outputs/active/${dataset}_${seed}|" "${temp_yaml}"

    log_file="logs/log_same_${dataset}_${seed}.txt"

    python src/rl/main.py "${temp_yaml}" # >> "${log_file}" 2>&1
    rm "${temp_yaml}"
  done
done