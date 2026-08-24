import numpy as np
import torch
from typing import Literal

# %% SPLIT FUNCTIONS
def split_idx_pretrain(I,
                       test_subject: int|None=None,
                       output: Literal['array', 'tensor']='array',
                       val_ratio=0.1):
    """
    Pre-train set을 subject마다 ``val_ratio``를 반영하여 train set과 validation set으로 나눔:
        - 기본적으로 test set은 두지 않지만, ``test_subject`` 설정 시, 그 사람의 sample을 test set으로 구분.
        - train (0), val (1)
        - test (2); **only if ``test_subject`` is not None**


    Parameters
    ----------
    I: np.ndarray|torch.Tensor
    test_subject: int|None
        if not None, set test set
    val_ratio: float
        The proportion of the validation set relative to the entire sample (default: 0.1)

    Returns
    ----------
    np.ndarray|torch.Tensor
        ``split_indices``: (NDArray[np.int32]|Tensor)

            train (0), val (1) (, test (2); **only if ``test_subject`` is not None**)

    """
    set_ids = np.unique(I)
    sample_nums = {sub: (I==sub).sum() for sub in set_ids}
    # -----
    if test_subject is None:
        test_subject = -100
    all_idx = []
    for sub in set_ids:
        samplenum = sample_nums[sub]
        if sub == test_subject:
            idxs = np.full(samplenum, 2)    # test set
        else:       # train & validation set
            idxs = np.zeros(samplenum)
            val_num = int(val_ratio * samplenum)
            valtest_idx = np.random.choice(samplenum, val_num, replace=False)
            idxs[valtest_idx] = 1

        all_idx.append(idxs)
    all_idx = np.concatenate(all_idx)
    if output == 'array':
        return all_idx.astype(dtype=np.int32)
    elif output == 'tensor':
        return torch.as_tensor(all_idx, dtype=torch.int32)
    else: raise ValueError(f'unexpected `output`: {output}')



def get_nest_cv_split_indices(I,
                              output:Literal['array', 'tensor']='array'):
    """

    Parameters
    ----------
    I: np.ndarray|torch.Tensor

    Returns
    -------
    (np.ndarray|torch.LongTensor, dict)
        ``nest_cv_split_indices``: np.ndarray|torch.LongTensor
            shape (#outer folds, #inner folds, #samples)

        ``nest_cv_test_val_subs``: dict
            key: test_sub_id; value: list[validation_sub_ids]

    """
    set_ids = sorted(np.unique(I).tolist())
    assert output in ['array', 'tensor']

    nest_cv_split_indices = []
    nest_cv_test_val_subs = {}

    for test_sub in np.unique(I, sorted=True):
        test_sub = int(test_sub)
        train_val_subs = [*set(set_ids).difference([test_sub])]
        train_val_subs = sorted(train_val_subs)

        nest_cv_test_val_subs[test_sub] = train_val_subs

        nest_split_indices = []
        for val_sub in train_val_subs:
            indices = np.zeros(len(I))
            indices[I==test_sub] = 2
            indices[I==val_sub] = 1

            nest_split_indices.append(indices)

        nest_split_indices = np.stack(nest_split_indices)
        nest_cv_split_indices.append(nest_split_indices)
    nest_cv_split_indices = np.stack(nest_cv_split_indices)
    if output == 'tensor':    nest_cv_split_indices = torch.LongTensor(nest_cv_split_indices)

    return nest_cv_split_indices, nest_cv_test_val_subs


# %% DATA FUNCTION
def prepare_data(data_fpath_fmt, set_ids, included_fband, return_I= False, **kwargs):
    """

    Parameters
    ----------
    data_fpath_fmt: str
    set_ids: typing.Sequence[int]
    included_fband: typing.Sequence[int]
    return_I: bool
    kwargs: Any
        used for ``data_fpath_fmt`` formatting

    Returns
    -------
    (torch.Tensor, torch.Tensor)|torch.Tensor
        ``X_time``: torch.Tensor
            shape (-1, #bands, #channels, #timepoints)
        ``I``: torch.Tensor
            1-D array of subject IDs (-1,).
            Return if ``return_I`` is True.
    """
    included_fband = np.array(included_fband, dtype=np.int32)
    X_time, I = [], []
    for sub in set_ids:
        X = np.load(data_fpath_fmt.format(subid=sub, **kwargs))[:, included_fband]  # (-1, F, N, T)
        # ---- EA
        X_time.append(X)
        # X_time.append(EAforTime(X, output='real'))    # X has already applied EA
        if return_I: I.append(torch.full((X_time[-1].shape[0],), sub))
    X_time = torch.Tensor(np.concatenate(X_time))     # (-1, F, N, T)
    # -----
    if return_I:
        I = torch.cat(I)
        return X_time, I
    return X_time


def prepare_SMC_RS(
        data_fpath_fmt,
        set_ids,
        included_fband=None,
        return_I=False,
        **kwargs    ):
    """
    Load SMC dataset.

    Parameters
    ----------
    data_fpath_fmt: str
    set_ids: typing.Sequence[int]
    included_fband: typing.Sequence[int]
    return_I: bool
    kwargs: Any
        used for ``data_fpath_fmt`` formatting

    Returns
    -------
    (torch.Tensor, torch.Tensor)|torch.Tensor
        ``X_time``: torch.Tensor
            shape (-1, #bands, #channels, #timepoints)
        ``I``: torch.Tensor
            1-D array of subject IDs (-1,).
            Return if ``return_I`` is True.

    """
    assert 'TSU_RS' not in data_fpath_fmt
    assert '/timeseries' in data_fpath_fmt

    if included_fband is not None:
        assert 'fband' in data_fpath_fmt
        included_fband = np.array(included_fband, dtype=np.int32)
    # -----
    X_time, I = [], []
    for sub in set_ids:
        if included_fband is None: X = np.load(data_fpath_fmt.format(subid=sub, **kwargs))
        else: X = np.load(data_fpath_fmt.format(subid=sub, **kwargs))[:, included_fband]
        # X = EAforTime(X, output='real')   # X has already applied EA
        X_time.append(X)
        if return_I: I.append(torch.full((X_time[-1].shape[0],), sub))
    X_time = torch.Tensor(np.concatenate(X_time))   # (-1, N, T) | (-1, F, N, T)

    if return_I: return X_time, torch.cat(I)
    return X_time


def prepare_TSU_RS(
        data_fpath_fmt,
        set_ids,
        included_fband=None,
        **kwargs    ):
    """
    Load TSU dataset.

    Parameters
    ----------
    data_fpath_fmt: str|typing.Sequence[str]
    set_ids: typing.Sequence[int]
    included_fband: typing.Sequence[int]|None
    kwargs: Any
        used for ``data_fpath_fmt`` formatting

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        ``X_time``: torch.Tensor
            shape (-1, #bands, #channels, #timepoints)
        ``I``: torch.Tensor
            shape (-1,)

    """
    if isinstance(data_fpath_fmt, str): data_fpath_fmt = [data_fpath_fmt]
    assert all([('TSU_RS' in fpath) for fpath in data_fpath_fmt])

    if included_fband is not None:
        included_fband = np.array(included_fband, dtype=np.int32)
    else:
        assert all([('/timeseries' in fpath) for fpath in data_fpath_fmt])
    # -----
    X_time, I = [], []
    for sub in set_ids:
        if included_fband is None: X = np.concatenate([np.load(fpath.format(subid=sub, **kwargs))
                                                       for fpath in data_fpath_fmt])
        else: X = np.concatenate([np.load(fpath.format(subid=sub, **kwargs))[:, included_fband]
                                  for fpath in data_fpath_fmt])
        # ----- EA
        # if apply_ea: X = EAforTime(X, output='real')      # X has already applied EA
        X_time.append(X)
        I.append(torch.full((X.shape[0],), sub))

    return torch.Tensor(np.concatenate(X_time)), torch.cat(I)     # (-1, N, T) | (-1, F, N, T)


def load_target_data(
        dataset_fpath_fmt,
        set_ids,
        n_samples,
        y_trues,
        included_fband=None,
        data_function=None,
        data_function_kws=None,
        return_I=False,
        verbose=True,
        **kwargs
):
    """
    Load time-series EEG signals (F-Bands or others) for **CLASSIFICATION** from several datasets.


    Parameters
    ----------
    dataset_fpath_fmt: str
        **Euclidean-space Alignment was already applied for each subject**
    set_ids: typing.Sequence[int]
        e.g., np.arange(1, 20 + 1, dtype=np.int32)
    n_samples: typing.Sequence[int]
        A 1-D array containing the number of samples for each subject
    y_trues: typing.Sequence[int]
        A 1-D array containing true target value (0/1) for each subject
    included_fband: typing.Sequence[int]|None=None
        array (List/NDArray/Tuple ...) of included F-Band indices. (e.g. [1, 2, 3] (theta, alpha, beta))
    data_function: typing.Callable|None
        data load function.
        It must include arguments of ['dataset_fpath_fmt', 'set_ids', 'return_I'].
        If None, use ``prepare_SMC_RS`` function.
    data_function_kws: dict|None
        optional arguments passed to ``data_function``
    return_I: bool
        if True, return tensor ``I`` (subject ID array for all samples)
    verbose: bool
        if True, display summary of process
    kwargs: Any
        used for ``data_fpath_fmt`` formatting


    Returns
    -------
    (np.ndarray, torch.Tensor, torch.Tensor)|(np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor)
        ``set_ids``: NDArray[np.int32]
            subject ID array for all samples

        ``X_time``: Tensor
            shape (-1, F, N, T) or others

        ``Y``: Tensor
            shape (-1,)

        ``I``: Tensor; Optional
            subject ID array for all samples. shape (-1,)
    """
    assert len(set_ids) == len(n_samples) == len(y_trues)

    ##### Load Input Data #####
    if data_function is None:
        data = prepare_SMC_RS(
            dataset_fpath_fmt, set_ids=set_ids, included_fband=included_fband, return_I=return_I, **kwargs)
    else:
        if data_function_kws is None: data_function_kws = {}
        data = data_function(dataset_fpath_fmt, set_ids, return_I=return_I, **data_function_kws)
    # -----
    if not return_I:
        X = data
        if verbose:
            print(f"📁 LOAD DATASET ({dataset_fpath_fmt})\n\t>> X:", 'x'.join([str(s) for s in X.shape]))
            print("-" * 75)
    else:
        X, I = data
        I = torch.as_tensor(I, dtype=torch.int32)
        if verbose:
            print(f"📁 LOAD DATASET ({dataset_fpath_fmt})\n\t>> X:", ' x '.join([str(s) for s in X.shape]))
            print(f"\t>> I: ", I.shape[0], f", (# of subjects: {len(torch.unique(I))}; ",
                  ','.join(map(str, [sub.item() for sub in torch.unique(I)[:3]])), ", ...)", sep='')
            print("-" * 75)
    ##### Load Target Data #####
    Y = torch.cat([
        torch.full((n_sample,), y_true, dtype=torch.int32)
        for n_sample, y_true in zip(n_samples, y_trues)    ]).squeeze()

    ##### RETURN #####
    if return_I: return set_ids, X, Y, I
    return set_ids, X, Y


def load_pretrain_data(
        dataset_fpath_fmt,
        set_ids,
        included_fband=None,
        data_function=None,
        data_function_kws=None,
        verbose=True,
        **kwargs
):
    """
    Load time-series EEG signals (F-Bands or others) for **SAMPLE RECONSTRUCTION** from several datasets.



    Parameters
    ----------
    dataset_fpath_fmt: str|typing.Sequence[str]
        **Euclidean-space Alignment was applied for each subject**.
    set_ids: typing.Sequence[int]
        e.g., np.arange(1, 22 + 1, dtype=np.int32)
    included_fband: typing.Sequence[int]|None
        array (List/NDArray/Tuple ...) of included F-Band indices. (e.g. [1, 2, 3] (theta, alpha, beta))
    data_function: typing.Callable|None
        data load function.
        It must include arguments of ['dataset_fpath_fmt', 'set_ids']
        If None, use ``prepare_TSU_RS`` function.
    data_function_kws: dict|None
        optional arguments passed to ``data_function`` (default: None)
    verbose: bool
        if True, display summary of process
    kwargs: Any
        used for ``data_fpath_fmt`` formatting


    Returns
    -------
    (np.ndarray, torch.Tensor, torch.Tensor)
        ``set_ids``: NDArray[np.int32]
            subject ID array for all samples.
            shape (-1,)
        ``X_time``: Tensor
            shape (-1, F, N, T) or others
        ``I``: torch.Tensor
            shape (-1,)
    """

    ##### Load Data #####
    if data_function is None:
        X, I = prepare_TSU_RS(
            dataset_fpath_fmt=dataset_fpath_fmt, set_ids=set_ids, included_fband=included_fband, **kwargs)
    else:
        if data_function_kws is None: data_function_kws = {}
        X, I = data_function(dataset_fpath_fmt=dataset_fpath_fmt, set_ids=set_ids, **data_function_kws)
    # -----
    if verbose:
        print("📁 LOAD DATASET")
        print("\t>> X:", 'x'.join([str(s) for s in X.shape]))
        print("-" * 75)
    ##### RETURN #####
    return set_ids, X, I

