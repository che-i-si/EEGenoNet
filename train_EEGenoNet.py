import random, time, os, argparse, numpy as np, pandas as pd, pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from backbones.EEGenoNet import EEGenoNet
from trainers.trainer_EEGenoNet import Trainer
from utils import get_nest_cv_split_indices, load_target_data
from easydict import EasyDict as edict

MODEL_TAG = 'EEGenoNet'
TARGET_DATASET = 'Stroke_MT'
CLASS_LABELs = [ "Val/Val",    "Met" ]
DATA_SUB_IDs = list(range(1, 20+1))
DATA_N_SAMPLEs = [ 95, 85, 81, 80, 83, 64, 62, 55, 56, 61, 60, 59, 58, 55, 66, 61, 41, 61, 41, 61 ]
DATA_Y_TRUEs = [ 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1 ]

GENED_SAMPLE_PATH = Path("./checkpoints/EEGenoNet_pretrain/gened_samples/BDNFbi_1/testSub{subid:02d}_valSub{valsub:02d}_{cls}.npy")
# (Up-sampling) After performing sample generation first, load the model during training


# %% SETTING
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
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gened_fpath_fmt', type=str, default=str(GENED_SAMPLE_PATH))
    parser.add_argument('--pretrained_model_savedir', type=str, default=None)
    parser.add_argument('--data_load_fpath_fmt', type=str, required=True,
                        help="e.g., 'DATA_DIR/Sub{subid:02d}.npy'")
    parser.add_argument('--device', type=str, default='cuda:0')

    return parser.parse_args()


def get_clf_args(
        data_load_fpath_fmt:str,
        gened_fpath_fmt:str|Path,
        pretrained_model_savedir:Path|str|None=None,
        n_classes:int|None=None,
        make_dir:bool=True,
        **kwargs    ) -> edict:

    gened_fpath_fmt:Path                = Path(gened_fpath_fmt)
    if pretrained_model_savedir is None:
        pretrained_model_savedir:Path   = gened_fpath_fmt.parents[2]
    elif isinstance(pretrained_model_savedir, str):
        pretrained_model_savedir:Path   = Path(pretrained_model_savedir)

    ##### LOAD ARGS #####
    with open(pretrained_model_savedir / "args_ae.pkl", 'rb') as f: args_ae = pickle.load(f)
    args_ae.update({'PRETRAIN_MODEL_SAVE_DIR': args_ae.model_savedir, 'PRETRAIN_MODEL_NAME': args_ae.model_name,})
    args = edict(args_ae.copy())
    ##### SETTING #####
    wktime = time.strftime('%y_%m_%d_%H_%M', time.localtime(time.time()))
    model_savedir = f'./checkpoints/{MODEL_TAG}/{wktime}/'
    if make_dir: os.makedirs(model_savedir, exist_ok=False)
    ##### UPDATE #####
    if n_classes is None: n_classes = len(np.unique(DATA_Y_TRUEs))
    args.update({
        'target_dataset': TARGET_DATASET,
        'data_load_fpath_fmt': data_load_fpath_fmt,
        'gened_fpath_fmt': str(gened_fpath_fmt),
        'n_minor_upsample': 'major', 'random_state': 0,
        'apply_clf_head': True, 'n_classes': n_classes,
        ##### TRAINING #####
        'clf_epochs': 100,
        'clf_lr': 1e-3,
        'clf_weight_decay': 5e-4,
        'clf_min_run_epochs': 10,
        'clf_early_stopping_patience': 5,
        ##### MODULES #####
        'dconv_bias': True,
        'dconv_dropout': 0.2,
        'k': 3,
        'clfDec_conv_layers': [{"in_channels": 1, "out_channels": 5, "kernel_size": (args.N, 1), "bias": True}, ],
        'clfDec_linear_layers': None,
        'clf_batchnorm': True,
        'clf_dropout': 0.5,
        'clf_activation': 'softmax',
        ##### PATHS #####
        'pretrained_config_fpath': str(gened_fpath_fmt.parents[2]/'pretrained_configs/pretrained.pt'),
        'model_savedir': model_savedir,
        'model_name': MODEL_TAG,
        'config_fpath_fmt': model_savedir+'trained_configs/testSub{subid:02d}_valSub{valsub:02d}.pt',
    })
    args.update(kwargs)
    return args


# %% MAIN
def main(args:edict):
    model_savedir = Path(args.model_savedir)

    # ----- SAVE ARGS
    with open(model_savedir / "args.pkl", 'wb') as f:   pickle.dump(args, f)
    print(f"""
[{args.model_name}]
>> MODEL_SAVEDIR: {args.model_savedir}
""")
    # SEC: Prepare dataset
    set_ids, X, Y, I = load_target_data(
        dataset_fpath_fmt=args.data_load_fpath_fmt,
        set_ids=DATA_SUB_IDs,
        n_samples=DATA_N_SAMPLEs,
        y_trues=DATA_Y_TRUEs,
        included_fband=args.included_fband,
        return_I=True, verbose=True )
    # ----- Split
    nest_cv_split_indices, nest_cv_test_val_subs = get_nest_cv_split_indices(I, output='array')  # (20, 19, 2524)

    # SEC: Nested-LOSO
    all_test_prob, all_test_pred, all_test_acc = [], [], {}

    for (testsub, train_val_subs), nest_split_indices in zip(nest_cv_test_val_subs.items(), nest_cv_split_indices):
        # `nest_split_indices`: (19, 2524)
        test_target = int(Y[nest_split_indices[1] == 2][0])
        print("\n"+'═' * 80)
        print(f"Test Subject {testsub:02d} ({CLASS_LABELs[test_target]})")
        print("\n"+'═' * 80)

        each_testsub_probs, each_testsub_preds, each_testsub_accs = [], [], []

        for val_sub, indices in zip(train_val_subs, nest_split_indices):
            trainset = TensorDataset(X[indices == 0], Y[indices == 0])
            valset = TensorDataset(X[indices == 1], Y[indices == 1])
            testset = TensorDataset(X[indices == 2], Y[indices == 2])
            print(f"\n── Validation: Sub{val_sub:02d}", "─" * 59)

            ##### LOAD GENERATED SAMPLES ######
            n_train_samples = np.array([int((Y[indices == 0] == cls).sum())
                                        for cls in Y[indices == 0].unique()])
            major_cls, n_major = int(n_train_samples.argmax()), int(n_train_samples.max())
            # ----- aligned with the number of samples in the major-class
            n_minor_upsample = np.array([n_major] * args.n_classes, dtype=int)
            n_add_samples = n_minor_upsample - n_train_samples
            # ----- load minor-class samples (Binary class)
            minor_cls = int(n_train_samples.argmin())
            fpath = args.gened_fpath_fmt.format(subid=int(testsub), valsub=int(val_sub), cls=minor_cls)
            print(">>> LOAD GENERATED SAMPLES: ", fpath)
            X_up = torch.Tensor(np.load(fpath)[:int(n_add_samples[minor_cls])])
            y_up = torch.full(size=(X_up.size(0),), fill_value=minor_cls, dtype=torch.float32)
            trainset = torch.utils.data.ConcatDataset(
                [trainset, TensorDataset(X_up, y_up)]   )
            print("⏩ UP-SAMPLED TRAIN SET: ",
                  ', '.join(
                      [f"({int(cls)}) {n_train_samples[cls]}->{n_train_samples[cls] + int((y_up == cls).sum())}" for cls
                       in map(int, Y.unique())]))
            del X_up, y_up

            ##### PREPARE DATASET #####
            trainset = DataLoader(trainset, args.batch_size, shuffle=True)
            valset = DataLoader(valset, args.batch_size, shuffle=False)
            testset = DataLoader(testset, args.batch_size, shuffle=False)
            ##### LOAD PRE-TRAINED STATE DICT #####
            model = EEGenoNet(args)
            ae_sdict = torch.load(args.pretrained_config_fpath, map_location='cpu')
            ae_sdict = {k.replace('Encoder.Encoder', 'Encoder'): v for k, v in ae_sdict.items()}
            model.load_state_dict(ae_sdict, strict=False)
            ##### TRAIN #####
            trainer = Trainer(
                model=model, args=args, device=args.device,
                save_path=args.config_fpath_fmt.format(subid=int(testsub), valsub=int(val_sub)), )
            trainer.fit(trainset, valset, save=True)
            ##### TEST #####
            _, test_acc, test_probs = trainer.test(testset, verbose=True, return_pred=True, pred_argmax=False)
            test_preds = test_probs.argmax(axis=-1)

            each_testsub_accs.append(test_acc)
            each_testsub_probs.append(test_probs)
            each_testsub_preds.append(test_preds)

            print("─" * 80)

        print(f"Test Subject {testsub:02d} ({CLASS_LABELs[test_target]})")
        print(f"― Mean ACC (%)              : {each_testsub_accs.mean():.02f} ({each_testsub_accs.std():.02f})")

        all_test_acc[testsub] = each_testsub_accs
        all_test_prob.append(each_testsub_probs)
        all_test_pred.append(each_testsub_preds)

    # SEC: End of all 20x19 tests
    all_test_acc = pd.DataFrame(all_test_acc).T
    all_test_acc.index.name = "TestSub"
    all_test_acc['Mean (SD)'] = all_test_acc.apply(lambda x: f"{x.mean():.02f} ({x.std():.02f})", axis=1)

    all_test_prob = np.concatenate(all_test_prob, axis=1)  # (19, 2524, 2)
    all_test_pred = np.concatenate(all_test_pred, axis=1)  # (19, 2524)


    # SEC: Save Results
    np.save(model_savedir / "nest_cv_split_indices.npy", nest_cv_split_indices)
    np.save(model_savedir / "all_test_prob.npy", all_test_prob)
    np.save(model_savedir / "all_test_pred.npy", all_test_pred)

    with pd.ExcelWriter(model_savedir / 'performance.xlsx') as writer:
        all_test_acc.to_excel(writer, sheet_name='all_test_acc')


# %% RUN
if __name__=='__main__':
    ps = get_parser()
    args = get_clf_args(
        data_load_fpath_fmt=ps.data_load_fpath_fmt,
        gened_fpath_fmt=ps.gened_fpath_fmt,
        device=ps.device)
    main(args)