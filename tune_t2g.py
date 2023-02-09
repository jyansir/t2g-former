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

from bin import T2GFormer
from lib import Transformations, build_dataset, prepare_tensors, DATA, make_optimizer

DATASETS = [
    'gesture', 'churn', 'eye', 'california', 
    'house', 'adult', 'otto', 'helena', 'jannis', 
    'higgs-small', 'fb-comments', 'year'
]

IMPLEMENTED_MODELS = [T2GFormer]

AUC_FOR_BINCLASS = False # if True use AUC metric for binclass task

def get_training_args():
    MODEL_CARDS = [x.__name__ for x in IMPLEMENTED_MODELS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='configs/')
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--model", type=str, default='T2GFormer', choices=MODEL_CARDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=16, help='default early stop epoch is 16 for DNN tuning')
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
device = torch.device('cuda')
args = get_training_args()
seed_everything(args.seed)

"""Datasets and Dataloaders"""
dataset_name = args.dataset
T_cache = True
normalization = args.normalization if args.normalization != '__none__' else None
transformation = Transformations(normalization=normalization)
dataset = build_dataset(DATA / dataset_name, transformation, T_cache)

if AUC_FOR_BINCLASS and dataset.is_binclass:
    metric = 'roc_auc' # AUC (binclass)
else:
    metric = 'score' # RMSE (regression) or ACC (classification)

if dataset.X_num['train'].dtype == np.float64:
    dataset.X_num = {k: v.astype(np.float32) for k, v in dataset.X_num.items()}

d_out = dataset.n_classes or 1
X_num, X_cat, ys = prepare_tensors(dataset, device=device)

if dataset.task_type.value == 'regression':
    y_std = ys['train'].std().item()

batch_size_dict = {
    'churn': 128, 'eye': 128, 'gesture': 128, 'california': 256, 'house': 256, 'adult': 256 , 
    'higgs-small': 512, 'helena': 512, 'jannis': 512, 'otto': 512, 'fb-comments': 512,
    'covtype': 1024, 'year': 1024, 'santander': 1024, 'microsoft': 1024, 'yahoo': 256}
val_batch_size = 1024 if args.dataset in ['santander', 'year', 'microsoft'] else 256 if args.dataset in ['yahoo'] else 8192
if args.dataset == 'epsilon':
    batch_size = 16 if args.dataset == 'epsilon' else 128 if args.dataset == 'yahoo' else 256
elif args.dataset not in batch_size_dict:
    if dataset.n_features <= 32:
        batch_size = 512
        val_batch_size = 8192
    elif dataset.n_features <= 100:
        batch_size = 128
        val_batch_size = 512
    elif dataset.n_features <= 1000:
        batch_size = 32
        val_batch_size = 64
    else:
        batch_size = 16
        val_batch_size = 16
else:
    batch_size = batch_size_dict[args.dataset]

num_workers = 0
data_list = [X_num, ys] if X_cat is None else [X_num, X_cat, ys]
train_dataset = TensorDataset(*(d['train'] for d in data_list))
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)
val_dataset = TensorDataset(*(d['val'] for d in data_list))
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers
)
test_dataset = TensorDataset(*(d['test'] for d in data_list))
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers
)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

"""Models"""
model_cls = eval(args.model)
n_num_features = dataset.n_num_features
cardinalities = dataset.get_category_sizes('train')
n_categories = len(cardinalities)
cardinalities = None if n_categories == 0 else cardinalities

"""Loss Function"""
loss_fn = (
    F.binary_cross_entropy_with_logits
    if dataset.is_binclass
    else F.cross_entropy
    if dataset.is_multiclass
    else F.mse_loss
)

"""utils function"""
def apply_model(model, x_num, x_cat=None):
    if any(issubclass(eval(args.model), x) for x in IMPLEMENTED_MODELS):
        return model(x_num, x_cat)
    else:
        raise NotImplementedError

@torch.inference_mode()
def evaluate(model, parts):
    model.eval()
    predictions = {}
    for part in parts:
        assert part in ['train', 'val', 'test']
        predictions[part] = []
        for batch in dataloaders[part]:
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )
            predictions[part].append(apply_model(model, x_num, x_cat))
        predictions[part] = torch.cat(predictions[part]).cpu().numpy()
    prediction_type = None if dataset.is_regression else 'logits'
    return dataset.calculate_metrics(predictions, prediction_type)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, optimizer):
    """Training"""
    n_epochs = 10000
    best_score = -np.inf
    no_improvement = 0
    EARLY_STOP = args.early_stop
    for epoch in range(1, n_epochs + 1):
        model.train()
        for iteration, batch in enumerate(train_loader):
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )
            optimizer.zero_grad()
            loss = loss_fn(apply_model(model, x_num, x_cat), y)
            loss.backward()
            optimizer.step()

        scores = evaluate(model, ['val'])
        val_score = scores['val'][metric]
        if val_score > best_score:
            best_score = val_score
            print(' <<< BEST VALIDATION EPOCH')
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement == EARLY_STOP:
            break
    return best_score

const_params = {
    "activation": "reglu",
    "initialization": "kaiming",
    "n_heads": 8,
    "prenormalization": True,
    "residual_dropout": 0.0,
}

# claim large datasets
large_datasets = ['year', 'covtype', 'microsoft']

if args.dataset in large_datasets:
    const_params['d_ffn_factor'] = 4/3

const_training_params = {
    'batch_size': batch_size,
    'eval_batch_size': val_batch_size,
    'optimizer': 'adamw',
    'patience': args.early_stop
}

def needs_wd(name):
    return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

# small learning rate for column embeddings 
# for statble topology learning process
def needs_small_lr(name):
    return any(x in name for x in ['.col_head', '.col_tail'])

def objective(trial):
    attention_dropout = trial.suggest_uniform('attention_dropout', 0, 0.5)
    d_ffn_factor = (
        4 / 3
        if args.dataset in large_datasets
        else trial.suggest_uniform('d_ffn_factor', 2/3, 8/3)
    )
    d_token = (
        trial.suggest_int('d_token', 64, 256, 8)
        if args.dataset not in large_datasets
        else trial.suggest_int('d_token', 64, 512, 8)
    )
    ffn_dropout = trial.suggest_uniform('ffn_dropout', 0, 0.5)
    n_layers = (
        trial.suggest_int('n_layers', 3, 4)
        if args.dataset not in large_datasets
        else trial.suggest_int('n_layers', 1, 2)
    )
    col_lr = trial.suggest_loguniform('col_lr', 5e-3, 5e-2) # learning rate for column embeddings
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    lr = (
        trial.suggest_loguniform('lr', 3e-5, 3e-4)
        if args.dataset in large_datasets
        else trial.suggest_loguniform('lr', 1e-5, 1e-3)
    )

    cfg_model = {
        **const_params,
        'attention_dropout': attention_dropout,
        'd_ffn_factor': d_ffn_factor,
        'd_token': d_token,
        'ffn_dropout': ffn_dropout,
        'n_layers': n_layers,
    }
    """set default"""
    cfg_model.setdefault('kv_compression', None)
    cfg_model.setdefault('kv_compression_sharing', None)
    cfg_model.setdefault('token_bias', True)
    # default FR-Graph settings
    cfg_model.setdefault('sym_weight', True)
    cfg_model.setdefault('sym_topology', False)
    cfg_model.setdefault('nsi', True)

    """prepare model arguments"""
    cfg_model = {
        # task related
        'd_numerical': n_num_features,
        'categories': cardinalities,
        'd_out': d_out,
        **cfg_model
    }
    model = model_cls(**cfg_model).to(device)

    """Optimizers"""
    for x in ['tokenizer', '.norm', '.bias']:
        assert any(x in a for a in (b[0] for b in model.named_parameters()))
    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k) and not needs_small_lr(k)]
    parameters_with_slr = [v for k, v in model.named_parameters() if needs_small_lr(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
    optimizer = make_optimizer(
        const_training_params['optimizer'],
        (
            [
                {'params': parameters_with_wd},
                {'params': parameters_with_slr, 'lr': col_lr, 'weight_decay': 0.0},
                {'params': parameters_without_wd, 'weight_decay': 0.0},
            ]
        ),
        lr,
        weight_decay,
    )

    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    
    best_val_score = train(model, optimizer)
    return best_val_score



cfg_file = f'{args.output}/cfg-tmp.json'
def save_per_iter(study, trial):
    saved_model_cfg = {**const_params}
    saved_training_cfg = {**const_training_params}
    for k in ['attention_dropout', 'd_ffn_factor', 'd_token', 'ffn_dropout', 'n_layers']:
        if k not in saved_model_cfg:
            saved_model_cfg[k] = study.best_trial.params.get(k)
    for k in ['col_lr', 'weight_decay', 'lr']:
        saved_training_cfg[k] = study.best_trial.params.get(k)
    
    hyperparams = {
        'eval_score': study.best_trial.value,
        'n_trial': study.best_trial.number,
        'dataset': args.dataset,
        'normalization': args.normalization,
        'model': saved_model_cfg,
        'training': saved_training_cfg,
    }
    with open(cfg_file, 'w') as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)

iterations = 50 if args.dataset in large_datasets else 100
study = optuna.create_study(direction="maximize")
study.optimize(func=objective, n_trials=iterations, callbacks=[save_per_iter])


cfg_file = f'{args.output}/cfg.json'
for k in ['attention_dropout', 'd_ffn_factor', 'd_token', 'ffn_dropout', 'n_layers']:
    if k not in const_params:
        const_params[k] = study.best_params.get(k)
for k in ['weight_decay', 'lr', 'col_lr']:
    const_training_params[k] = study.best_params.get(k)

hyperparams = {
    'eval_score': study.best_value,
    'n_trial': study.best_trial.number,
    'dataset': args.dataset,
    'normalization': args.normalization,
    'model': const_params,
    'training': const_training_params,
}
with open(cfg_file, 'w') as f:
    json.dump(hyperparams, f, indent=4, ensure_ascii=False)