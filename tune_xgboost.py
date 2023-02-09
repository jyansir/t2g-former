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

import optuna

from xgboost import XGBClassifier, XGBRegressor
from lib import Transformations, build_dataset, prepare_tensors, DATA, make_optimizer


DATASETS = [
    'gesture', 'churn', 'eye', 'california', 
    'house', 'adult', 'otto', 'helena', 'jannis', 
    'higgs-small', 'fb-comments', 'year'
]

AUC_FOR_BINCLASS = False # if True use AUC metric for binclass task

def get_training_args():
    MODEL_CARDS = ["XGBoost"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='configs/')
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--model", type=str, default='XGBoost', choices=MODEL_CARDS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    args.output = str(Path(args.output) / f'{args.dataset}/{args.model}')
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    
    return args

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
args= get_training_args()
seed_everything(args.seed)

"""Datasets and Dataloaders"""
dataset_name = args.dataset
T_cache = True
transformation = Transformations(cat_encoding='one-hot') # especially for datasets with categorical X
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

# claim large datasets
large_datasets = ['year', 'covtype', 'microsoft']

model_param_space = {
    "alpha": (1e-8, 1e2, 'loguniform'),
    "booster": "gbtree",
    "colsample_bylevel": (0.5, 1, 'uniform'),
    "colsample_bytree": (0.5, 1, 'uniform'),
    "gamma": (1e-8, 1e2, 'loguniform'),
    "lambda": (1e-8, 1e2, 'loguniform'),
    "learning_rate": (1e-5, 1, 'loguniform'),
    "max_depth": (
        (3, 10, 'int')
        # if args.dataset not in large_datasets
        if args.dataset not in large_datasets
        else (6, 10, 'int')
    ),
    "min_child_weight": (1e-8, 1e5, 'loguniform'),
    "n_estimators": 2000, # default num of trees
    "n_jobs": -1,
    "subsample": (0.5, 1, 'uniform'),
    "tree_method": "gpu_hist"
}

running_time = 0.
def objective(trial):
    model_hyper = {}
    for key, value in model_param_space.items():
        if isinstance(value, tuple):
            model_hyper[key] = eval(f'trial.suggest_{value[-1]}')(key, *value[:-1])
        else:
            model_hyper[key] = value
    cfg = {
        "fit": {
            "early_stopping_rounds": 50,
            "verbose": False
        },
        "model": model_hyper,
    }
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

    global running_time
    start = time.time()
    model.fit(
        Xs['train'],
        dataset.y['train'],
        eval_set=[(Xs['val'], dataset.y['val'])],
        **cfg['fit'],
    )
    running_time += time.time() - start
    prediction = {'val': predict(Xs['val'])}
    prediction_type = None if dataset.is_regression else 'probs'
    scores = dataset.calculate_metrics(prediction, prediction_type)
    return scores['val'][metric]

cfg_file = f'{args.output}/cfg-tmp.json'
fit_cfg = {
    "early_stopping_rounds": 50,
    "verbose": False
}
def save_per_iter(study, trial):
    model_cfg = {}
    for key, value in model_param_space.items():
        if isinstance(value, tuple):
            model_cfg[key] = study.best_trial.params.get(key)
        else:
            model_cfg[key] = value
    
    hyperparams = {
        'time': running_time,
        'eval_score': study.best_trial.value,
        'n_trial': study.best_trial.number,
        'dataset': args.dataset,
        'fit': fit_cfg,
        'model': model_cfg,
    }
    if dataset.X_cat is not None:
        hyperparams['cat_encoding'] = transformation.cat_encoding
    with open(cfg_file, 'w') as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)
    
    if (trial.number + 1) % 100 == 0:
        extra_cfg_dir = f'{args.output}/{trial.number + 1}'
        if not os.path.exists(extra_cfg_dir):
            os.makedirs(extra_cfg_dir)
        with open(f'{extra_cfg_dir}/cfg.json', 'w') as f:
            json.dump(hyperparams, f, indent=4, ensure_ascii=False)

iterations = 100 # same as DNN
study = optuna.create_study(direction="maximize")
study.optimize(func=objective, n_trials=iterations, callbacks=[save_per_iter])


cfg_file = f'{args.output}/cfg.json'
model_cfg = {}
for key, value in model_param_space.items():
    if isinstance(value, tuple):
        model_cfg[key] = study.best_params.get(key)
    else:
        model_cfg[key] = value

hyperparams = {
    'time': running_time,
    'eval_score': study.best_trial.value,
    'n_trial': study.best_trial.number,
    'dataset': args.dataset,
    'cat_encoding': transformation.cat_encoding,
    'fit': fit_cfg,
    'model': model_cfg,
}

with open(cfg_file, 'w') as f:
    json.dump(hyperparams, f, indent=4, ensure_ascii=False)