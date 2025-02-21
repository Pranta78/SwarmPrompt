#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8  

BUDGET=10
POPSIZE=10
TEMPLATE=v1
initial=all
LLM_TYPE=deepseek

for dataset in sam
do
OUT_PATH=outputs/sum/$dataset/gpt/all/pso/bd${BUDGET}_top${POPSIZE}_para_topk_init/${TEMPLATE}/${LLM_TYPE}
for SEED in 5 10 15
do
python run.py \
    --seed $SEED \
    --dataset $dataset \
    --task sum \
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
    --cache_path data/sum/$dataset/prompts_auto.json \
    --template $TEMPLATE \
    --setting default \
    --output $OUT_PATH/seed$SEED
done
python get_result.py -p $OUT_PATH > $OUT_PATH/result.txt
done