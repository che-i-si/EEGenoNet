"""
Pretrain EEGenoNet Autoencoder model.

Notes
-----
Task: EEG sample reconstruction using the TSU dataset.

Model: EEGenoNet autoencoder (``backbones.EEGenoNet.EEGAutoEncoder``)
    - EEGenoNet Encoder + Reconstruction Decoder
"""

import numpy as np, pandas as pd, os, sys, json, pickle as pkl, time, random, argparse, yaml
from pathlib import Path

import torch
from torch import Tensor
from backbones.EEGenoNet import EEGAutoEncoder

from torch.utils.data import TensorDataset, DataLoader
from utils import load_pretrain_data, split_idx_pretrain
from trainers.trainer_EEGenoNet_AE import Trainer
from easydict import EasyDict as edict

from typing import Sequence, Literal, List, Dict


##### SETTINGS #####
MODEL_TAG   = 'EEGenoNet_pretrain'
MODEL_CONFIG_PATH = Path("model_config.yaml")

PRETRAIN_DATASET = 'TSU_RS'
DATA_SUB_IDs = list(range(1, 22+1))
INCLUDED_FBANDs = [ 1, 2, 3 ]   # theta, alpha, beta

# -----
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
os.environ["PYTHONHASHSEED"] = "0"
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.RandomState(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.set_float32_matmul_precision('medium')


# %% ARGS
def get_ae_args(
    data_load_fpath_fmt: str|List[str],
    make_dir: bool = True,
    device='cuda:0',
    **kwargs    ):

    with open(MODEL_CONFIG_PATH, 'r') as f: model_config = yaml.load(f, Loader=yaml.FullLoader)
    args = edict({
        ##### DATASET #####
        'included_fband': INCLUDED_FBANDs,
        'data_load_fpath_fmt': data_load_fpath_fmt,
        ##### MODEL #####
        **model_config,
        'apply_clf_head': False,

        ##### TRAINING #####
        "ae_val_ratio": 0.1,
        "ae_epochs": 200,
        "ae_lr": 0.001,
        'batch_size': 32,
        "ae_weight_decay": 0,
        "ae_min_run_epochs": 50,
        "ae_early_stopping_patience": 10,

        ##### SETTINGS #####
        'device': device,
    })
    args.update(kwargs)
    ##### SETTING
    # wktime = time.strftime('%y_%m_%d_%H_%M', time.localtime(time.time()))
    # model_savedir = PRJ_DIR+f'/checkpoints/{MODEL_TAG}/{wktime}'
    model_savedir = f'checkpoints/{MODEL_TAG}/'
    if make_dir: os.makedirs(model_savedir, exist_ok=True)

    args.update({
        'model_name': MODEL_TAG,
        'pretrained_config_fpath': model_savedir+'pretrained_configs/pretrained.pt',
        'model_savedir': model_savedir,
    })
    return args



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_load_fpath_fmt', nargs='+', type=str, required=True,
                        help="e.g., 'DATA_DIR/EC/Sub{subid:02d}.npy' 'DATA_DIR/EO/Sub{subid:02d}.npy'")
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device (default: %(default)s)")

    return parser.parse_args()


# %% TRAIN
def main(args:edict):
    model_savedir   = Path(args.model_savedir)
    # ----- SAVE ARGS
    with open(model_savedir / "ae_args.pkl", 'wb') as f:   pkl.dump(args, f)

    ##### LOAD DATASET #####
    set_ids, X, I   = load_pretrain_data(
        dataset_fpath_fmt=args.data_load_fpath_fmt, set_ids=DATA_SUB_IDs,
        included_fband=args.included_fband, verbose=True    )

    # ------ Data split
    split_indices = split_idx_pretrain(I, output='array', val_ratio=args.ae_val_ratio)

    ##### PREPARE DATASET #####
    trainset    = DataLoader(TensorDataset(X[split_indices==0]), batch_size=args.batch_size, shuffle=True)
    valset      = DataLoader(TensorDataset(X[split_indices == 1]), batch_size=args.batch_size, shuffle=False)

    ##### TRAIN #####
    model       = EEGAutoEncoder(args)
    trainer     = Trainer(
        model, args=args, save_path=args.pretrained_config_fpath, device=args.device)
    trainer     .fit(trainset, valset)
    ##### SAVE #####
    np.save(model_savedir / 'indices.npy', split_indices)
    return args


# %% RUN
if __name__ == '__main__':
    ps      = get_parser()
    args    = get_ae_args(
        data_load_fpath_fmt=ps.data_load_fpath_fmt,
        make_dir=True,
        device=ps.device    )
    main(args)
