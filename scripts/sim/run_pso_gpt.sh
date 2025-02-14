#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  

POPSIZE=10
BUDGET=10
template=v1
initial=all
LLM_TYPE=gpt4

for dataset in asset
do
OUT_PATH=outputs/sim/$dataset/gpt/$initial/pso/bd${BUDGET}_top${POPSIZE}_topk_para_init/$template/${LLM_TYPE}_10_samples
for SEED in 5 10 15
do
python run.py \
    --seed $SEED \
    --dataset $dataset \
    --task sim \
    --batch-size 20 \
    --prompt-num 0 \
    --sample_num 10 \
    --language_model gpt \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --position pre \
    --evo_mode pso \
    --llm_type $LLM_TYPE \
    --initial $initial \
    --initial_mode para_topk \
    --template $template \
    --cache_path data/sim/$dataset/prompts.json \
    --output $OUT_PATH/seed${SEED}
done
python get_result.py -p $OUT_PATH > $OUT_PATH/result.txt
done