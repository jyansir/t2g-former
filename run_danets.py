import argparse
import os
import json
import random
import numpy as np
from pathlib import Path
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch.distributed
import torch.backends.cudnn
from sklearn.metrics import accuracy_score, mean_squared_error
from qhoptim.pyt import QHAdam
from bin.danet.config.default import cfg

from lib import Transformations, build_dataset, prepare_tensors, DATA, make_optimizer
from bin.danet.DAN_Task import DANetClassifier, DANetRegressor

DATASETS = [
    'gesture', 'churn', 'eye', 'california', 
    'house', 'adult', 'otto', 'helena', 'jannis', 
    'higgs-small', 'fb-comments', 'year'
]

AUC_FOR_BINCLASS = False

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

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch v1.4, DANet Task Training')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument("--output", type=str, default='results/')
    parser.add_argument("--n_layers", type=int, default=28, choices=[20, 24, 28])
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.model = f'DANets-{args.n_layers}'
    args.output = str(Path(args.output) / f'{args.model}/{args.dataset}/{args.seed}')
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    args.config = f'bin/danet/config/{args.dataset}-{args.n_layers}.yaml' # pre-defined configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if args.config:
        cfg.merge_from_file(args.config)
    cfg.seed = args.seed
    cfg.freeze()
    task = cfg.task
    seed = cfg.seed
    train_config = {'dataset': cfg.dataset, 'resume_dir': cfg.resume_dir, 'logname': cfg.logname}
    fit_config = dict(cfg.fit)
    model_config = dict(cfg.model)
    print('Using config: ', cfg)

    return args, train_config, fit_config, model_config, task, seed, len(args.gpu_id)

def set_task_model(task, std=None, seed=1):
    if task == 'classification':
        clf = DANetClassifier(
            optimizer_fn=QHAdam,
            optimizer_params=dict(lr=fit_config['lr'], weight_decay=1e-5, nus=(0.8, 1.0)),
            scheduler_params=dict(gamma=0.95, step_size=20),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            layer=model_config['layer'],
            base_outdim=model_config['base_outdim'],
            k=model_config['k'],
            drop_rate=model_config['drop_rate'],
            seed=seed
        )
        eval_metric = ['accuracy']

    elif task == 'regression':
        clf = DANetRegressor(
            std=std,
            optimizer_fn=QHAdam,
            optimizer_params=dict(lr=fit_config['lr'], weight_decay=1e-5, nus=(0.8, 1.0)),
            scheduler_params=dict(gamma=0.95, step_size=20),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            layer=model_config['layer'],
            base_outdim=model_config['base_outdim'],
            k=model_config['k'],
            seed=seed
        )
        eval_metric = ['rmse']
    return clf, eval_metric

if __name__ == '__main__':

    """args"""
    print('===> Setting configuration ...')
    args, train_config, fit_config, model_config, task, seed, n_gpu = get_args()
    logname = None if train_config['logname'] == '' else train_config['dataset'] + '/' + train_config['logname']
    print('===> Getting data ...')
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
    X_num, X_cat, ys = dataset.X_num, dataset.X_cat, dataset.y
    X = {k: X_num[k] if X_cat is None else np.concatenate((X_cat[k], X_num[k]), axis=1) for k in ys.keys()}
    cat_idxs = [] if X_cat is None else list(range(X_cat['train'].shape[1]))
    if dataset.is_regression:
        ys = {k: v.reshape(-1, 1) for k, v in ys.items()}

    """Models"""
    n_num_features = dataset.n_num_features
    cardinalities = dataset.get_category_sizes('train')
    n_categories = len(cardinalities)
    
    std = dataset.y_info['std'] if dataset.is_regression else None

    clf, eval_metric = set_task_model(task, std, seed)

    danet_metric = 'rmse' if dataset.is_regression else 'accuracy'

    clf.fit(
        X_train=X['train'], y_train=ys['train'],
        eval_set=[(X['val'], ys['val'])],
        eval_name=['val'],
        eval_metric=eval_metric,
        max_epochs=fit_config['max_epochs'], patience=fit_config['patience'],
        batch_size=fit_config['batch_size'], virtual_batch_size=fit_config['virtual_batch_size'],
        logname=logname,
        resume_dir=train_config['resume_dir'],
        n_gpu=n_gpu
    )

    val_metrics = clf.history[f"val_{danet_metric}"]
    losses = clf.history['loss']

    y_pred = clf.predict(X['test'])
    prediction_type = 'logits' if dataset.is_binclass else None
    final_test_score = dataset.calculate_metrics({'test': y_pred}, prediction_type)['test'][metric] # wrong AUC for binary classification

    """Record Exp Results"""
    record_exp(
        final_test_score,
        losses=str(losses), val_score=str(val_metrics),
        cfg=cfg,
    )
