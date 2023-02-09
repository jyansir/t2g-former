import os
import math
import time
import json
import random
import argparse
import numpy as np
from pathlib import Path
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from xgboost import XGBClassifier, XGBRegressor
from lib import Transformations, build_dataset, prepare_tensors, DATA, make_optimizer


DATASETS = [
    'gesture', 'churn', 'eye', 'california', 
    'house', 'adult', 'otto', 'helena', 'jannis', 
    'higgs-small', 'fb-comments', 'year'
]

AUC_FOR_BINCLASS = False # if True use AUC metric for binclass task

def get_training_args():
    MODEL_CARDS = ['XGBoost']
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='results/')
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--model", type=str, default='XGBoost', choices=MODEL_CARDS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg_file = f'configs/{args.dataset}/{args.model}/cfg.json'
    try:
        print(f"Try to load configuration file: {cfg_file}")
        with open(cfg_file, 'r') as f:
            cfg = json.load(f)
    except IOError as e:
        print(f"Not exist !")
        cfg_file = f'configs/default/{args.model}/cfg.json'

        print(f"Try to load default configuration")
        assert os.path.exists(cfg_file), f'Please give a default configuration file: {cfg_file}'
        with open(cfg_file, 'r') as f:
            cfg = json.load(f)
    
    args.output = str(Path(args.output) / f'{args.model}/{args.dataset}/{args.seed}')
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    
    return args, cfg


def record_exp(final_score, **kwargs):
    results = {
        'config': vars(args),
        'final': final_score,
        **kwargs,
    }
    exp_list = [file for file in os.listdir(args.output) if '.json' in file]
    exp_list = [int(file.split('.')[0]) for file in exp_list]
    exp_id = 0 if len(exp_list) == 0 else max(exp_list) + 1
    with open(f"{args.output}/{exp_id}.json", 'w', encoding='utf8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    

def seed_everything(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


"""args"""
args, cfg = get_training_args()
seed_everything(args.seed)

"""Datasets and Dataloaders"""
dataset_name = args.dataset
T_cache = True
transformation = Transformations(cat_encoding='one-hot')
dataset = build_dataset(DATA / dataset_name, transformation, T_cache)

if AUC_FOR_BINCLASS and dataset.is_binclass:
    metric = 'roc_auc' # AUC (binclass)
else:
    metric = 'score' # RMSE (regression) or ACC (classification)

if dataset.X_num['train'].dtype == np.float64:
    dataset.X_num = {k: v.astype(np.float32) for k, v in dataset.X_num.items()}

if dataset.X_cat is None:
    Xs = {k: dataset.X_num[k] for k in ['train', 'val', 'test']}
else:
    Xs = {k: np.concatenate((dataset.X_num[k], dataset.X_cat[k]), axis=1) for k in ['train', 'val', 'test']}

if dataset.is_regression:
    model = XGBRegressor(**cfg['model'], random_state=args.seed)
    predict = model.predict
else:
    model = XGBClassifier(
        **cfg['model'], random_state=args.seed, disable_default_eval_metric=True
    )
    if dataset.is_multiclass:
        predict = model.predict_proba
        cfg['fit']['eval_metric'] = 'merror'
    else:
        predict = lambda x: model.predict_proba(x)[:, 1]
        cfg['fit']['eval_metric'] = 'error'

model.fit(
    Xs['train'],
    dataset.y['train'],
    eval_set=[(Xs['val'], dataset.y['val'])],
    **cfg['fit'],
)

prediction = {k: predict(v) for k, v in Xs.items()}
prediction_type = None if dataset.is_regression else 'probs'
scores = dataset.calculate_metrics(prediction, prediction_type)
for k in scores:
    print(k, scores[k][metric])

"""Record Exp Results"""
record_exp(
    scores['test'][metric],
    train_score=scores['train'][metric], val_score=scores['val'][metric],
    cfg=cfg, metric=scores,
)
