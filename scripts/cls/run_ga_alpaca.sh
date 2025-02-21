#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  
export CUDA_VISIBLE_DEVICES=0

BUDGET=10
POPSIZE=10
SEED=5
GA=topk
LLM_TYPE=turbo

for dataset in sst2-5
do
OUT_PATH=outputs/cls/$dataset/alpaca/all/ga/bd${BUDGET}_top${POPSIZE}_para_topk_init/$GA/$LLM_TYPE
for SEED in 15
do
python run.py \
    --seed $SEED \
    --dataset $dataset \
    --task cls \
    --batch-size 32 \
    --prompt-num 0 \
    --sample_num 500 \
    --language_model gpt \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --position demon \
    --evo_mode ga \
    --llm_type $LLM_TYPE \
    --setting default \
    --initial all \
    --initial_mode para_topk \
    --ga_mode $GA \
    --cache_path data/cls/$dataset/seed${SEED}/prompts_batched.json \
    --output $OUT_PATH/seed$SEED \
    --dev_file ./data/cls/$dataset/seed${SEED}/dev.txt
done
python get_result.py -p $OUT_PATH > $OUT_PATH/result.txt
done