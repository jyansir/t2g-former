#!/bin/bash -i

OUTPUT='configs/'
PATIENCE=16
MODELS=('DCNv2' 'AutoInt' 'MLP' 'NODE' 'SNN' 'FTTransformer')
DATASETS=('gesture' 'churn' 'eye' 'california' 'house' 'adult' 'otto' 'helena' 'jannis' 'higgs-small' 'fb-comments' 'year')
NORMS=('quantile' 'quantile' 'standard' 'quantile' 'quantile' 'quantile' '__none__' 'standard' 'quantile' 'quantile' 'quantile' 'quantile')

for i in "${!DATASETS[@]}"; do

    # other baselines
    for MODEL in ${MODELS[@]}; do
        python tune_baselines.py \
            --output $OUTPUT \
            --model $MODEL \
            --dataset ${DATASETS[$i]} \
            --normalization ${NORMS[$i]} \
            --early_stop $PATIENCE
    done

    # T2G-Former
    python tune_t2g.py \
        --output $OUTPUT \
        --dataset ${DATASETS[$i]} \
        --normalization ${NORMS[$i]} \
        --early_stop $PATIENCE
        
    # XGBoost
    python tune_xgboost.py \
        --output $OUTPUT \
        --seed $SEED \
        --dataset ${DATASETS[$i]}
done
