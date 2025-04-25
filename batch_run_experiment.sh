#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

datasets=(agnews sst-2 winowhy epistemic_reasoning trec amazon hyperbaton timedial aqua)
# datasets=(trec amazon hyperbaton timedial aqua)
# strategies=(random max-entropy best-of-k global-entropy-ordering)
strategies=(random max-entropy)
seeds=(1241 352 107694 215 745)  
models=(
  # "meta-llama/Meta-Llama-3-8B"
  # "meta-llama/Llama-3.1-8B"
  # "meta-llama/Llama-3.2-3B"
  # "Qwen/Qwen2.5-7B"
  "Qwen/Qwen2.5-3B"
  # "meta-llama/Meta-Llama-3-8B"
)
mkdir -p logs
#####################
for model_name in "${models[@]}"; do
  # create a filesystem‐safe version of the model name
  model_safe=$(echo "$model_name" | tr '/.' '__')
  for dataset in "${datasets[@]}"; do
    for strategy in "${strategies[@]}"; do
      for seed in "${seeds[@]}"; do

        temp_yaml="prompting_configs/temp_${dataset}_${strategy}_${seed}.yaml"
        
        cp prompting_configs/baseline-gpt2.yaml "${temp_yaml}"
        
        sed -i "s/^task: .*/task: ${dataset}/" "${temp_yaml}"
        sed -i "s/^seed: .*/seed: ${seed}/" "${temp_yaml}"
        sed -i "s/^strategy: .*/strategy: ${strategy}/" "${temp_yaml}"

        sed -i "s|^model_name: .*|model_name: ${model_name}|" "$temp_yaml"

        output_dir="./outputs/${model_safe}/${strategy}_${dataset}_${seed}"
        sed -i "s|^output_dir: .*|output_dir: ${output_dir}|" "$temp_yaml"
        mkdir -p "${output_dir}"
        
        log_file="logs/log_${model_safe}_${dataset}_${strategy}_${seed}.txt"
        python src/prompting/main.py "$temp_yaml" >> "$log_file" 2>&1
        
        rm "${temp_yaml}"
      done
    done
  done
done