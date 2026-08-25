"""
Generate samples using the pre-trained EEGenoNet Autoencoder model (up-sampling)

"""

import numpy as np, os, argparse, random, pickle
from pathlib import Path
import torch
from backbones.EEGenoNet import EEGAutoEncoder
from utils import load_target_data
from itertools import product


##### MANUAL SETTINGS #####
MODEL_TAG = 'EEGenoNet_pretrain'

PRETRAIN_MODEL_SAVEDIR = Path("checkpoints") / MODEL_TAG
PRETRAIN_CONFIG_PATH = PRETRAIN_MODEL_SAVEDIR / 'pretrained_configs/pretrained.pt'

TARGET_DATASET = 'Stroke_MT'
CLASS_LABELs = [ "Val/Val",    "Met" ]
DATA_SUB_IDs = list(range(1, 20+1))
DATA_N_SAMPLEs = [ 95, 85, 81, 80, 83, 64, 62, 55, 56, 61, 60, 59, 58, 55, 66, 61, 41, 61, 41, 61 ]
DATA_Y_TRUEs = [ 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1 ]

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


# %% UTILS
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model_savedir', type=str, default=str(PRETRAIN_MODEL_SAVEDIR))
    parser.add_argument('--pretrained_config_fpath', type=str, default=str(PRETRAIN_CONFIG_PATH))
    parser.add_argument('--data_load_fpath_fmt', type=str, required=True,
                        help="e.g., 'DATA_DIR/Sub{subid:02d}.npy'")
    parser.add_argument('--device', type=str, default='cuda:0')

    return parser.parse_args()

# %% GENERATION FUNCTION
def gen_samples(
        pretrain_model_savedir:str|Path,
        pretrained_config_fpath:str|Path,
        data_load_fpath_fmt: str|Path,
        n_gen_per_cls:int,
        batch_size:int=32,
        device: str='cuda:0',
):
    pretrain_model_savedir = Path(pretrain_model_savedir)
    pretrained_config_fpath = Path(pretrained_config_fpath)
    gened_sample_save_dir = pretrain_model_savedir / "gened_samples"
    gened_sample_save_fpath_fmt = str(gened_sample_save_dir / 'testSub{subid:02d}_valSub{valsub:02d}_{cls}.npy')
    gened_sample_save_dir.mkdir(parents=True, exist_ok=True)

    with open(pretrain_model_savedir / 'args_ae.pkl', 'rb') as f: args_ae = pickle.load(f)
    assert args_ae.model_name == MODEL_TAG

    ##### LOAD DATASET #####
    set_ids, X, Y, I = load_target_data(
        dataset_fpath_fmt=str(data_load_fpath_fmt),
        set_ids=DATA_SUB_IDs,
        n_samples=DATA_N_SAMPLEs,
        y_trues=DATA_Y_TRUEs,
        included_fband=args_ae.included_fband,
        return_I=True, verbose=True )

    ##### LOAD MODEL #####
    model = EEGAutoEncoder(args_ae)
    model.load_state_dict(torch.load(pretrained_config_fpath, map_location='cpu'), strict=True)
    model.to(device)
    model.eval()
    print("📁 LOAD PRE-TRAINED MODEL FROM: ", str(pretrained_config_fpath))
    ##### LOSO #####
    for testsub, valsub in product(set_ids, set_ids):
        if testsub == valsub: continue
        print(f"""
───── Test Subject: {testsub:02d} | Validation Subject: {valsub:02d} ───────────────""")
        X_train, Y_train, I_train = X[(I!=testsub)&(I!=valsub)], Y[(I!=testsub)&(I!=valsub)], I[(I!=testsub)&(I!=valsub)]
        for cls in Y_train.unique():
            xx, yy, ii = X_train[Y_train==cls], Y_train[Y_train==cls], I_train[I_train==cls]
            ##### TOKEN 생성 #####
            temp_tokens, spat_tokens, spec_tokens = [], [], []
            for start_idx in torch.arange(0, xx.shape[0], batch_size):
                temp_token, spat_token, spec_token = model.Encoder(xx[start_idx:start_idx+batch_size].to(device), get_token=True)
                temp_tokens.append(temp_token.detach().cpu())
                spat_tokens.append(spat_token.detach().cpu())
                spec_tokens.append(spec_token.detach().cpu())
            temp_tokens, spat_tokens, spec_tokens = torch.cat(temp_tokens), torch.cat(spat_tokens), torch.cat(spec_tokens)
            ##### TOKEN SHUFFLE & UP-SAMPLE #####
            indices = torch.randint(0, temp_tokens.size(0), size=(3, n_gen_per_cls))
            temp_tokens, spat_tokens, spec_tokens = temp_tokens[indices[0]], spat_tokens[indices[1]], spec_tokens[indices[2]]
            ##### BASIS MATRIX #####
            basis = torch.einsum('bfd, bdn, bdt -> bfnt', spec_tokens, temp_tokens, spat_tokens).to(device)
            del temp_tokens, spat_tokens, spec_tokens
            ##### RECONSTRUCTION #####
            xx = []
            for start_idx in torch.arange(0, n_gen_per_cls, batch_size):
                xx.append(
                    model.outLayer(
                        torch.cat([block(basis[start_idx:start_idx+batch_size, [f_i]]) for f_i, block in enumerate(model.Decoder)], dim=1),
                    ).detach().cpu().numpy()
                )
            xx = np.concatenate(xx, axis=0)
            ##### SAVE #####
            gened_sample_save_fpath = gened_sample_save_fpath_fmt.format(subid=testsub, valsub=valsub, cls=cls)
            print(f"[{CLASS_LABELs[int(cls)]}]\tX_gen: ({' x '.join([str(s) for s in xx.shape])})")
            print(f"\t\t◽", gened_sample_save_fpath)
            np.save(gened_sample_save_fpath, xx)
        print()


if __name__ == '__main__':
    ps = get_parser()
    gen_samples(
        pretrain_model_savedir=ps.pretrain_model_savedir,
        pretrained_config_fpath=ps.pretrained_config_fpath,
        data_load_fpath_fmt=ps.data_load_fpath_fmt,
        n_gen_per_cls=3000,
        device=ps.device)



