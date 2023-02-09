#!/bin/bash -i

OUTPUT='results/'
PATIENCE=16
MODELS=('DCNv2' 'AutoInt' 'MLP' 'NODE' 'SNN' 'FTTransformer')
SEEDS=(42 20 31 53 64 75 86 97 108 119 130 141 152 163 174)
DATASETS=('gesture' 'churn' 'eye' 'california' 'house' 'adult' 'otto' 'helena' 'jannis' 'higgs-small' 'fb-comments' 'year')
NORMS=('quantile' 'quantile' 'standard' 'quantile' 'quantile' 'quantile' '__none__' 'standard' 'quantile' 'quantile' 'quantile' 'quantile')

for SEED in ${SEEDS[@]}; do
    for i in "${!DATASETS[@]}"; do

        # other baselines
        for MODEL in ${MODELS[@]}; do
            python run_baselines.py \
                --output $OUTPUT \
                --seed $SEED \
                --model $MODEL \
                --dataset ${DATASETS[$i]} \
                --normalization ${NORMS[$i]} \
                --early_stop $PATIENCE
        done

        # T2G-Former
        python run_t2g.py \
            --output $OUTPUT \
            --seed $SEED \
            --dataset ${DATASETS[$i]} \
            --normalization ${NORMS[$i]} \
            --early_stop $PATIENCE \
            --froze_later
        
        # DANets-28
        python run_danets.py \
            --output $OUTPUT \
            --seed $SEED \
            --dataset ${DATASETS[$i]} \
            --normalization ${NORMS[$i]} \
            --n_layers 28
        
        # TF TabNet
        python run_tf_tabnet.py \
            --output $OUTPUT \
            --seed $SEED \
            --dataset ${DATASETS[$i]} \
            --normalization ${NORMS[$i]} \
            --early_stop $PATIENCE
        
        # XGBoost
        python run_xgboost.py \
            --output $OUTPUT \
            --seed $SEED \
            --dataset ${DATASETS[$i]}
    done
done