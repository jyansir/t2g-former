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

from bin import DCNv2, AutoInt, MLP, NODE, ResNet, SNN
from bin import FTTransformer
from lib import Transformations, build_dataset, prepare_tensors, DATA, make_optimizer

DATASETS = [
    'gesture', 'churn', 'eye', 'california', 
    'house', 'adult', 'otto', 'helena', 'jannis', 
    'higgs-small', 'fb-comments', 'year'
]

IMPLEMENTED_MODELS = [DCNv2, AutoInt, MLP, NODE, ResNet, SNN, FTTransformer]

AUC_FOR_BINCLASS = False


def get_training_args():
    MODEL_CARDS = [x.__name__ for x in IMPLEMENTED_MODELS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='results/')
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CARDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=16, help='early stopping for finetune')
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


def record_exp(final_score, best_score, **kwargs):
    results = {
        'config': vars(args),
        'final': final_score,
        'best': best_score,
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
device = torch.device('cuda')
args, cfg = get_training_args()
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
"""set default"""
if args.model in ['MLP', 'ResNet', 'SNN', 'NODE']:
    cfg['model'].setdefault('d_embedding', None)
if args.model in ['FTTransformer', 'AutoInt']:
    cfg['model'].setdefault('kv_compression', None)
    cfg['model'].setdefault('kv_compression_sharing', None)
if args.model == 'FTTransformer':
    cfg['model'].setdefault('token_bias', True)
"""prepare model arguments"""
if args.model in ['AutoInt', 'ResNet', 'FTTransformer']:
    kwargs = {
        'd_numerical': n_num_features,
        'd_out': d_out,
        'categories': cardinalities,
        **cfg['model']
    }
elif args.model in ['DCNv2', 'MLP', 'SNN', 'NODE']:
    kwargs = {
        'd_in': n_num_features,
        'd_out': d_out,
        'categories': cardinalities,
        **cfg['model']
    }
else:
    raise NotImplementedError
model = model_cls(**kwargs).to(device)

"""Optimizers"""
if args.model in ['AutoInt', 'FTTransformer']:
    def needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

    for x in ['tokenizer', '.norm', '.bias']:
        assert any(x in a for a in (b[0] for b in model.named_parameters()))
    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
    optimizer = make_optimizer(
        cfg['training']['optimizer'],
        (
            [
                {'params': parameters_with_wd},
                {'params': parameters_without_wd, 'weight_decay': 0.0},
            ]
        ),
        cfg['training']['lr'],
        cfg['training']['weight_decay'],
    )
elif args.model in ['DCNv2', 'MLP', 'ResNet', 'SNN', 'NODE']:
    optimizer = make_optimizer(
        cfg['training']['optimizer'],
        model.parameters(),
        cfg['training']['lr'],
        cfg['training']['weight_decay'],
    )
else:
    raise NotImplementedError

if torch.cuda.device_count() > 1:  # type: ignore[code]
    print('Using nn.DataParallel')
    model = nn.DataParallel(model)

"""Loss Function"""
loss_fn = (
    F.binary_cross_entropy_with_logits
    if dataset.is_binclass
    else F.cross_entropy
    if dataset.is_multiclass
    else F.mse_loss
)

"""utils function"""
def apply_model(x_num, x_cat=None):
    if any(issubclass(eval(args.model), x) for x in IMPLEMENTED_MODELS):
        return model(x_num, x_cat)
    else:
        raise NotImplementedError

@torch.inference_mode()
def evaluate(parts):
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
            predictions[part].append(apply_model(x_num, x_cat))
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



"""Training"""
init_score = evaluate(['test'])['test'][metric]
print(f'Test score before training: {init_score: .4f}')

losses, val_metric, test_metric = [], [], []
n_epochs = 10000
report_frequency = len(ys['train']) // batch_size // 3
loss_holder = AverageMeter()
best_score = -np.inf
final_test_score = -np.inf
best_test_score = -np.inf
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
        loss = loss_fn(apply_model(x_num, x_cat), y)
        loss.backward()
        optimizer.step()
        loss_holder.update(loss.item(), len(ys))
        if iteration % report_frequency == 0:
            print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss_holder.val:.4f} (avg_loss) {loss_holder.avg:.4f}')
    losses.append(loss_holder.avg)
    loss_holder.reset()

    scores = evaluate(['val', 'test'])
    val_score, test_score = scores['val'][metric], scores['test'][metric]
    val_metric.append(val_score), test_metric.append(test_score)
    print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')
    if val_score > best_score:
        best_score = val_score
        final_test_score = test_score
        print(' <<< BEST VALIDATION EPOCH')
        no_improvement = 0
    else:
        no_improvement += 1
    if test_score > best_test_score:
        best_test_score = test_score
    if no_improvement == EARLY_STOP:
        break

"""Record Exp Results"""
record_exp(
    final_test_score, best_test_score,
    losses=str(losses), val_score=str(val_metric), test_score=str(test_metric),
    cfg=cfg,
)
