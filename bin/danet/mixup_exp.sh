#!/bin/bash -i


DATASETS=('california' 'helena' 'jannis' 'adult' 'churn' 'gesture')
NORMS=('quantile' 'standard' 'quantile' 'quantile' 'quantile' 'quantile')
SEEDS=(20 31 53 64 75)
for SEED in ${SEEDS[@]}; do
    for i in "${!DATASETS[@]}"; do
        python main.py \
            --seed $SEED \
            --dataset ${DATASETS[$i]} \
            --normalization ${NORMS[$i]} \
            --gpu_id 1 \
            --n_layers 28 \
            --mixup
    done
done