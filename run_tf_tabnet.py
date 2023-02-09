"""
TensorFlow Runtime Environment

tensorflow-gpu: 1.14.0
CUDA: 11.6
GPU: GeForce RTX 2080 Ti
"""
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
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import TensorDataset, DataLoader
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from bin.tabnet import TabNet
from lib import Transformations, build_dataset, DATA

DATASETS = [
    'gesture', 'churn', 'eye', 'california', 
    'house', 'adult', 'otto', 'helena', 'jannis', 
    'higgs-small', 'fb-comments', 'year'
]

IMPLEMENTED_MODELS = [TabNet]


def make_tf_loaders(seed, e, b, X, Y):
    tf.set_random_seed(seed)
    datasets = {k: tf.data.Dataset.from_tensor_slices((X[k], Y[k])) for k in X.keys()}
    datasets = {k: tf.data.Dataset.from_tensor_slices((X[k], Y[k])) for k in X.keys()}
    X_loader = {}
    Y_loader = {}

    for k in datasets.keys():
        if k == 'train':
            datasets[k] = datasets[k].shuffle(
                buffer_size=50, reshuffle_each_iteration=True
            )
            datasets[k] = datasets[k].batch(
                b, drop_remainder=True
            )
        else:
            datasets[k] = datasets[k].batch(b)

        # NOTE +1 for the final validation step for the best model at the end
        datasets[k] = datasets[k].repeat(e + 1)
        datasets[k] = datasets[k].make_initializable_iterator()

        X_loader[k], Y_loader[k] = datasets[k].get_next()

    # Add train with no shuffle dataset for final eval
    ds = tf.data.Dataset.from_tensor_slices((X['train'], Y['train']))
    ds = ds.batch(b)
    ds = ds.make_initializable_iterator()
    k = "train_noshuffle"
    datasets[k] = ds
    X_loader[k], Y_loader[k] = ds.get_next()

    return datasets, X_loader, Y_loader


def get_train_eval_ops(cfg, data, model, x, y):
    "Create train step, train loss, val/test predict ops"
    encoder_out_train, total_entropy = model.encoder(
        x['train'], reuse=False, is_training=True
    )
    encoder_out_val, _ = model.encoder(x['val'], reuse=True, is_training=False)
    encoder_out_test, _ = model.encoder(x['test'], reuse=True, is_training=False)
    encoder_out_train_noshuffle, _ = model.encoder(
        x["train_noshuffle"], reuse=True, is_training=False
    )
    train_op = None

    # Regression and classification losses
    if data.is_multiclass:
        y_pred_train, _ = model.classify(encoder_out_train, reuse=False)
        y_pred_train_noshuffle, _ = model.classify(
            encoder_out_train_noshuffle, reuse=True
        )
        y_pred_val, _ = model.classify(encoder_out_val, reuse=True)
        y_pred_test, _ = model.classify(encoder_out_test, reuse=True)
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            y['train'], y_pred_train, reduction=tf.losses.Reduction.MEAN
        )
        train_loss_op = (
            cross_entropy + cfg["training"]["sparsity_loss_weight"] * total_entropy
        )
    if data.is_regression:
        y_pred_train = model.regress(encoder_out_train, reuse=False)
        y_pred_train_noshuffle = model.regress(encoder_out_train_noshuffle, reuse=True)
        y_pred_val = model.regress(encoder_out_val, reuse=True)
        y_pred_test = model.regress(encoder_out_test, reuse=True)
        mse = tf.losses.mean_squared_error(
            tf.expand_dims(y['train'], axis=1),
            y_pred_train,
            reduction=tf.losses.Reduction.MEAN,
        )
        train_loss_op = mse + cfg["training"]["sparsity_loss_weight"] * total_entropy
    if data.is_binclass:
        y_pred_train = model.regress(encoder_out_train, reuse=False)
        y_pred_train_noshuffle = model.regress(encoder_out_train_noshuffle, reuse=True)
        y_pred_val = model.regress(encoder_out_val, reuse=True)
        y_pred_test = model.regress(encoder_out_test, reuse=True)
        log_loss = tf.losses.log_loss(
            tf.expand_dims(y['train'], axis=1),
            tf.nn.sigmoid(y_pred_train),
            reduction=tf.losses.Reduction.MEAN,
        )
        train_loss_op = (
            log_loss + cfg["training"]["sparsity_loss_weight"] * total_entropy
        )

    # Optimization step
    global_step = tf.train.get_or_create_global_step()
    # learning_rate = tf.compat.v1.train.exponential_decay(
    #     global_step=global_step,
    #     **cfg["training"]["schedule"],
    # )
    learning_rate = tf.train.exponential_decay(
        global_step=global_step,
        **cfg["training"]["schedule"],
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        gvs = optimizer.compute_gradients(train_loss_op)
        grad_thresh = cfg["training"]["grad_thresh"]
        capped_gvs = [
            (tf.clip_by_value(grad, -grad_thresh, grad_thresh), var)
            for grad, var in gvs
        ]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    y_pred_op = {
        'train': y_pred_train_noshuffle,
        'val': y_pred_val,
        'test': y_pred_test,
    }

    return train_op, train_loss_op, y_pred_op


def evaluate(sess, y_pred_op, parts, idx_loaders):
    metrics = {}
    predictions = {}

    for part in parts:
        idx_loader = idx_loaders[part]
        y_pred = []
        for idx in idx_loader:
            if use_placeholders:
                feed_dict = {
                    x_loader[part]: X[part][idx],
                    y_loader[part]: Y[part][idx],
                }
                feed_dict[x_loader["train_noshuffle"]] = X['train'][idx]
                feed_dict[y_loader["train_noshuffle"]] = Y['train'][idx]
            else:
                feed_dict = None

            y_pred.append(sess.run(y_pred_op[part], feed_dict=feed_dict))
        y_pred = np.concatenate(y_pred)
        predictions[part] = y_pred
    prediction_type = None if dataset.is_regression else 'logits'
    metrics = dataset.calculate_metrics(predictions, prediction_type)

    return metrics, predictions


def get_training_args():
    MODEL_CARDS = [x.__name__ for x in IMPLEMENTED_MODELS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='results/')
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--model", type=str, default='TabNet', choices=MODEL_CARDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=16)
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
    tf.reset_default_graph()
    tf.set_random_seed(seed)




"""args"""
args, cfg = get_training_args()
seed_everything(args.seed)

"""Datasets and Dataloaders"""
dataset_name = args.dataset
T_cache = True
normalization = args.normalization if args.normalization != '__none__' else None
transformation = Transformations(normalization=normalization)
dataset = build_dataset(DATA / dataset_name, transformation, T_cache)

d_out = dataset.n_classes or 1
X_num, X_cat, Y = (None if x is None else x for x in [dataset.X_num, dataset.X_cat, dataset.y])
X = (X_num, X_cat)

batch_size_dict = {
    'churn': 128, 'eye': 128, 'gesture': 128, 'california': 256, 'house': 256, 'adult': 256 , 
    'higgs-small': 512, 'helena': 512, 'jannis': 512, 'otto': 512, 'fb-comments': 512,
    'covtype': 1024, 'year': 1024, 'santander': 1024, 'microsoft': 1024, 'yahoo': 256}
batch_size = cfg['training'].get('batch_size', batch_size_dict[args.dataset])


use_placeholders = dataset_name in ["epsilon"]
columns = None

if use_placeholders:
    X = X_num
    # Epsilon dataset doesn't work with tf data Dataset well
    x_loader = {
        k: tf.placeholder(tf.float32, shape=(None, dataset.n_num_features))
        for k in X.keys()
    }
    y_loader = {k: tf.placeholder(tf.int32, shape=(None,)) for k in Y.keys()}

    x_loader["train_noshuffle"] = tf.placeholder(
        tf.float32, shape=(None, dataset.n_num_features)
    )
    y_loader["train_noshuffle"] = tf.placeholder(tf.float32, shape=(None,))
else:
    if X_cat is not None:
        X = {}
        for part in ['train', 'val', 'test']:
            X[part] = {}
            for i in range(X_num[part].shape[1]):
                X[part][str(i)] = X_num[part][:, i]
            for i in range(
                X_num[part].shape[1], X_num[part].shape[1] + X_cat[part].shape[1]
            ):
                X[part][str(i)] = X_cat[part][:, i - X_num[part].shape[1]]
    else:
        X = X_num
    
    datasets_tf, x_loader, y_loader = make_tf_loaders(args.seed, cfg['training']['epochs'], batch_size,  X, Y)

    if X_cat is not None:
        num_columns = [
            tf.feature_column.numeric_column(str(i))
            for i in range(X_num['train'].shape[1])
        ]
        cat_columns = [
            tf.feature_column.categorical_column_with_identity(
                str(i), max(X_cat['train'][:, i - X_num['train'].shape[1]]) + 1
            )
            for i in range(
                X_num['train'].shape[1],
                X_num['train'].shape[1] + X_cat['train'].shape[1],
            )
        ]
        emb_columns = [
            tf.feature_column.embedding_column(c, cfg["model"]["d_embedding"])
            for c in cat_columns
        ]
        columns = num_columns + emb_columns

cfg['model']['output_dim'] = cfg['model']['feature_dim']
print(columns)

model = TabNet(
    num_classes=d_out,
    columns=columns,
    num_features=X_num['train'].shape[1]
    + (0 if X_cat is None else cfg["model"]["d_embedding"] * X_cat['train'].shape[1]),
    **cfg["model"],
)

train_op, train_loss_op, y_pred_op = get_train_eval_ops(
    cfg, dataset, model, x_loader, y_loader
)

init = tf.initialize_all_variables()
init_local = tf.local_variables_initializer()
init_table = tf.tables_initializer(name="Initialize_all_tables")

epoch_size = (
    Y['train'].shape[0] // batch_size
)  # drop_last=True in tf
saver = tf.train.Saver()

"""Index Loader"""
train_idx_dataset = TensorDataset(torch.arange(Y['train'].shape[0]))
train_idx_loader = DataLoader(
    dataset=train_idx_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
val_idx_dataset = TensorDataset(torch.arange(Y['val'].shape[0]))
val_idx_loader = DataLoader(
    dataset=val_idx_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
test_idx_dataset = TensorDataset(torch.arange(Y['test'].shape[0]))
test_idx_loader = DataLoader(
    dataset=test_idx_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
idx_loaders = {'train': train_idx_loader, 'val': val_idx_loader, 'test': test_idx_loader}


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
losses, val_metric, test_metric = [], [], []
loss_holder = AverageMeter()
best_score = -np.inf
final_test_score = -np.inf
best_test_score = -np.inf
no_improvement = 0
EARLY_STOP = args.early_stop

with tf.Session() as sess:
    sess.run(init)
    sess.run(init_local)
    sess.run(init_table)
    if not use_placeholders:
        for k in datasets_tf.keys():
            sess.run(datasets_tf[k].initializer)

    for e in range(cfg["training"]["epochs"]):
        for step, idx in enumerate(train_idx_loader):
            if use_placeholders:

                feed_dict = {
                    x_loader['train']: X['train'][idx],
                    y_loader['train']: Y['train'][idx],
                }
            else:
                feed_dict = None

            _, train_loss = sess.run([train_op, train_loss_op], feed_dict=feed_dict)
            loss_holder.update(train_loss, len(idx))
            if step % cfg["training"]["display_steps"] == 0:
                print(f"Step {step}, AVG Train Loss {loss_holder.avg:.4f}")

        losses.append(loss_holder.avg)
        loss_holder.reset()
        metrics, predictions = evaluate(
            sess, y_pred_op, ['val', 'test'], idx_loaders
        )
        val_score, test_score = metrics['val']['score'], metrics['test']['score']
        val_metric.append(val_score), test_metric.append(test_score)
        print(f'Epoch {e:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')

        if val_score > best_score:
            best_score = val_score
            final_test_score = test_score
            print(' <<< BEST VALIDATION EPOCH')
            no_improvement = 0
            # saver.save(sess, str(output / "checkpoint.ckpt"))
        else:
            no_improvement += 1
        if test_score > best_test_score:
            best_test_score = test_score
        if no_improvement == EARLY_STOP:
            print("Early stopping")
            break

    # saver.restore(sess, str(output / "checkpoint.ckpt"))
    # stats['metrics'], predictions = evaluate(
    #     args, sess, Y, y_pred_op, lib.PARTS, D.info['task_type'], y_info
    # )
    # for k, v in predictions.items():
    #     np.save(output / f'p_{k}.npy', v)
    # saver.save(sess, str(output / "best.ckpt"))


"""Record Exp Results"""
record_exp(
    final_test_score, best_test_score,
    losses=str(losses), val_score=str(val_metric), test_score=str(test_metric),
    cfg=cfg,
)
