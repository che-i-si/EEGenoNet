import torch, math
from torch import nn, Tensor
from backbones._utils import cal_cnn_outlen
from easydict import EasyDict as edict
from typing import Tuple, Dict, List, Sequence, Optional, Union, Literal
# -----
from backbones.blocks import Embedding, TemporalSpatialEncoder, Transpose, Interpolation, GELUTanh

# %% ENCODER
class Encoder(nn.Module):
    """
    (1) Patch Embedding: F-band wise Conv2d

    (2) Token Embedding:

        (2.1) Spectral Token Embedding: F-band wise [Conv2d (kernel: Nx1) + Linear (indim: outlen, outdim: 1)]

        (2.2) Bi-axial Embedding:
            (2.2.1) Interpolation: match the out time length on the highest f-band's

            (2.2.2) Dot-Conv2d: x (B x h x N x t) -...-> (B x d x N x t)

            (2.2.3) Bi-axial Embedding: positional embedding & add cls tokens & one former block

            (2.2.4) Former Blocks: former block 'nlayer' times

    Forward
    ----------
        **Input:**
            **x**: Tensor
                Input time-series EEG signals. shape (B x F x N x T)

        **Output:**
                **spec_token**: shape (B x F x d)

                **x**: shape (B x d x (1+N) x (1+t)

    """
    def __init__(self, F:int, N:int, T:int,
                 embdim:int,
                 patch_conv_layers:Sequence[Sequence[Dict]],
                 intp_mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-exact'],
                 dot_conv_outdims:Sequence[int],
                 nlayer:int, nhead:int,
                 enc_dropout: float|int):
        super().__init__()
        self.F, self.N, self.T, self.embdim = F, N, T, embdim
        assert F == len(patch_conv_layers)
        assert all([layers[-1]["out_channels"]==patch_conv_layers[0][-1]["out_channels"]
                    for layers in patch_conv_layers[1:]])   # 모든 fband-wise conv block의 outdim (=h; hiddim)이 동일
        ##### PATCH EMBEDDING #####
        self.patchEmbed = nn.ModuleList([self._conv2d_block(layers, enc_dropout)
                                              for layers in patch_conv_layers])         # F-Band 마다 temporal convolution
        self.out_lens = [cal_cnn_outlen(fband_conv, T, pos=1)
                         for fband_conv in self.patchEmbed]                        # F-Band 마다 convolution out len (=t1, t2, t3)
        self.outlen = self.out_lens[-1]      # 가장 고주파 대역의 out len
        self.hiddim = patch_conv_layers[-1][-1]["out_channels"]     # (=h)
        ##### SPECTRAL TOKEN #####
        self.specTokenEmbed = nn.ModuleList([
            nn.Sequential(nn.Conv2d(self.hiddim, embdim, (N, 1), bias=True),
                          # nn.BatchNorm2d(embdim),
                          nn.ReLU(),
                          nn.Dropout2d(enc_dropout),

                          nn.Flatten(start_dim=2),
                          nn.Linear(outlen, 1, bias=True),
                          # nn.ReLU(),
                          nn.Flatten(),)
            for outlen in self.out_lens
        ])
        ##### BIAXIAL TOKEN #####
        self.Intp = Interpolation((N, self.outlen), mode=intp_mode)
        self.biaxialEmbed = nn.ModuleList([
            self._conv2d_block(layers=[{"in_channels": self.hiddim if i==0 else dot_conv_outdims[i-1],
                                        "out_channels": outdim,
                                        "kernel_size": (1, 1),
                                        "bias": True}
                                       for i, outdim in enumerate(dot_conv_outdims)],
                               dropout=enc_dropout),
            Embedding(embdim=embdim, nhead=nhead, N=N, seq_len=self.outlen, dropout=0.1)
        ])
        self.formerBlocks = nn.ModuleList([
            TemporalSpatialEncoder(embdim=embdim, nhead=nhead,)
            for _ in range(nlayer)
        ])


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        (1) Patch Embedding: F-band wise Conv2d

        (2) Token Embedding:

            (2.1) Spectral Token Embedding: F-band wise [Conv2d (kernel: Nx1) + Linear (indim: outlen, outdim: 1)]

            (2.2) Bi-axial Embedding:
                (2.2.1) Interpolation: match the out time length on the highest f-band's

                (2.2.2) Dot-Conv2d: x (B x h x N x t) -...-> (B x d x N x t)

                (2.2.3) Bi-axial Embedding: positional embedding & add cls tokens & one former block

                (2.2.4) Former Blocks: former block 'nlayer' times

        Parameters
        ----------
        x: Tensor
            Input time-series EEG signals. shape (B x F x N x T)

        Returns
        -------
        Tuple[Tensor, Tensor]
            **spec_token**: shape (B x F x d)

            **x**: shape (B x d x (1+N) x (1+t)

        """
        assert [*x.shape[1:]]==[self.F, self.N, self.T]
        # (1) Patch Embedding
        x = [block(x[:, [f_i]]) for f_i, block in enumerate(self.patchEmbed)]
        # (2.1) Spectral Token Embedding
        spec_token = torch.stack([block(xx) for xx, block in zip(x, self.specTokenEmbed)], dim=1)   # spec_token (B x F x d)
        # (2.2) Bi-axial Token Embedding
        x = torch.stack([self.Intp(xx) for xx in x], dim=1).sum(dim=1)         # x (B x F x h x N x t) -> (B x h x N x t)
        for block in self.biaxialEmbed: x = block(x)        # x (B x d x N x t) -> (B x d x (1+N) x (1+t)
        for block in self.formerBlocks:
            x = x + block(x)            # x (B x (1+N) x d x (1+t))
        x = x.transpose(1, 2)           # x (B x d x (1+N) x (1+t)
        return spec_token, x

    ##############################
    def _conv2d_block(self, layers: Sequence[Dict], dropout: float|int):
        block = nn.Sequential()
        for layer in layers:
            block.extend(nn.Sequential(nn.Conv2d(**layer),
                                       Transpose([1, 3]),
                                       nn.LayerNorm(layer["out_channels"]),
                                       Transpose([1, 3]),
                                       nn.ReLU(),
                                       nn.Dropout2d(dropout)))
        return block

# %% DYCONV
class DyConv(nn.Module):
    """
    Dynamic Convolutional Block

    (1) **Spectral-Temporal Convolutional Block:**

        dynamic kernel shape: B x 1 x d x N x 1

        output feature (`x_st`) shape: B x 1 x N x t

    (2) **Spectral-Spatial Convolutional Block:**

        dynamic kernel shape: B x 1 x d x 1 x k

        output feature (`x_ss`) shape: B x 1 x N x t

    (3) **Temporal-Spatial Convolutional Block:**

        dynamic filter shape: B x 1 x d x N x t (positional convolution)

        output feature (`x_ts`) shape: B x 1 x N x t

    (4) **Summation:**
        `x_st` + `x_ss` + `x_ts`

    Forward
    ----------
    **Input:**
        <**spec_token**>: Tensor
            shape (B x F x d)
        <**x**>: Tensor
            shape (B x d x (1+N) x (1+t))

    **Output:**
        <**x**>: Tensor
            output feature of dynamic-convolution.
            shape (B x 1 x N x t)

    Attributes
    ----------
    F: int
        # of F-band
    N: int
        # of EEG channels
    T: int
        # of input EEG time samples
    outlen: int
        # of time-patches (=t)
    embdim: int
        # of embedding dimension
    k: int
        kernel size of temporal dynamic convolution
    specTempFGB: nn.Sequential
        Conv2d (d, d, (F, 1); ReLU) + Conv2d (d, d, (1, 1); ReLU)
    specSpatFGB: nn.Sequential
        Conv2d (d, d, (F, 1); ReLU) + Conv2d (d, d, (1, 5), stride=(1, 2); ReLU)
        + AdaptiveAvgPool2d((1, k))
    tempSpatFGB: nn.Sequential
        Conv2d (d, d, (1, 1); Sigmoid)
        + temp_spat_token (itself)
    unfold: nn.Sequential
        Unfold (kernel: (1, k), padding: (1, (k-1)//2))
        + Unflatten + Unflatten

    Parameters
    ----------
    F: int
        # of F-band
    N: int
        # of EEG channels
    T: int
        # of input EEG time samples
    outlen: int
        # of time-patches (=t)
    embdim: int
        # of embedding dimension
    k: int
        kernel size of temporal dynamic convolution
    dconv_bias: bool
        bias of temporal FGBs
    dconv_dropout: float
        dropout rate of FGBs

    """

    def __init__(self, F:int, N:int, T:int, outlen:int,
                 embdim:int, k: int,
                 dconv_bias:bool, dconv_dropout:float,
                 ):
        super().__init__()
        self.F, self.N, self.T, self.outlen, self.embdim = F, N, T, outlen, embdim
        self.k = k
        ##### FILTER GENERATION BLOCK #####
        self.specTempFGB = nn.Sequential(nn.Conv2d(embdim, embdim, (F, 1), bias=dconv_bias),
                                         # nn.BatchNorm2d(embdim),
                                         nn.ReLU(),
                                         nn.Dropout2d(dconv_dropout),       # (B x d x 1 x N)

                                         nn.Conv2d(embdim, embdim, (1, 1), bias=dconv_bias),
                                         nn.ReLU(),
                                         nn.Dropout2d(dconv_dropout),
                                         )
        self.specSpatFGB = nn.Sequential(nn.Conv2d(embdim, embdim, (F, 1), bias=dconv_bias),
                                         # nn.BatchNorm2d(embdim),
                                         nn.ReLU(),
                                         nn.Dropout2d(dconv_dropout),       # (B x d x 1 x N)

                                         nn.Conv2d(embdim, embdim, (1, 5), stride=(1, 2), bias=dconv_bias),
                                         nn.ReLU(),
                                         nn.Dropout(dconv_dropout),
                                         nn.AdaptiveAvgPool2d((1, k)),      # (B x d x 1 x k)
                                         )
        self.tempSpatFGB = nn.Sequential(nn.Conv2d(embdim, embdim, (1, 1), bias=dconv_bias),
                                         nn.Sigmoid(),
                                         nn.Dropout2d(dconv_dropout),       # (B x d x N x t)
                                         )
        self.unfold = nn.Sequential(nn.Unfold(kernel_size=(1, self.k), padding=(0, (self.k - 1) // 2)),     # (B x dk x Nt)
                                    nn.Unflatten(dim=1, unflattened_size=(self.embdim, self.k)),
                                    nn.Unflatten(dim=3, unflattened_size=(self.N, self.outlen)))            # (B x d x k x N x t)
    def forward(self, spec_token: Tensor, x: Tensor) -> Tensor:
        """

        Parameters
        ----------
        spec_token: Tensor
            shape (B x F x d)
        x: Tensor
            shape (B x d x (1+N) x (1+t))

        Returns
        -------
        Tensor
            **x**:
            output feature of dynamic-convolution.
            shape (B x 1 x N x t)

        """
        temp_token = x[:, :, 1:, 0]     # temp_token (B x d x N)
        spat_token = x[:, :, 0, 1:]     # spat_token (B x d x t)
        x = x[:, :, 1:, 1:]             # x (B x d x N x t)
        # (1) Spectral-Temporal
        x_st = self.specTempFGB(
            torch.einsum('bfd, bdn -> bdfn', spec_token, temp_token)
        )       # (B x d x F x N) -> (B x d x 1 x N)
        x_st = torch.einsum('bdnt, bdn -> bnt', x, x_st.flatten(2)).unsqueeze(1)                # (B x 1 x N x t)
        # (2) Spectral-Spatial
        x_ss = self.specSpatFGB(
            torch.einsum('bfd, bdt -> bdft', spec_token, spat_token)
        )       # (B x d x F x t) -> (B x d x 1 x k)
        x_ss = torch.einsum('bdknt, bdk -> bnt', self.unfold(x), x_ss.flatten(2)).unsqueeze(1)  # (B x 1 x N x t)
        # (3) Temporal-Spatial
        x_ts = torch.einsum('bdn, bdt -> bdnt', temp_token, spat_token)
        x_ts = self.tempSpatFGB(x_ts) + x_ts       # (B x d x N x t) -> (B x d x N x t)
        x_ts = torch.einsum(
            'bdknt, bdknt -> bnt', self.unfold(x), self.unfold(x_ts)
        ).unsqueeze(1)                                                                          # (B x 1 x N x t)
        # (4) Summation
        x = x_st + x_ss + x_ts                                                                  # x (B x 1 x N x t)
        return x

# %% CLASSIFIER
class ClassifierHead(nn.Module):
    def __init__(self, N: int, outlen: int,
                 clfDec_conv_layers: Sequence[Dict], clfDec_linear_layers: Sequence[Dict]|None,
                 n_classes: int,
                 clf_batchnorm: bool, clf_dropout: float,
                 clf_activation:Literal['sigmoid', 'softmax', 'logsigmoid', 'logsoftmax', 'none'],
                 ):
        super().__init__()
        self.N, self.outlen = N, outlen
        assert clfDec_conv_layers[0]["in_channels"] == 1
        ##### DECODER #####
        self.clfDec = self._conv2d_block(clfDec_conv_layers, clf_batchnorm, clf_dropout)
        featlen = clfDec_conv_layers[-1]["out_channels"] * cal_cnn_outlen(self.clfDec, N, pos=0) * cal_cnn_outlen(self.clfDec, outlen, pos=1)
        self.clfDec.append(nn.Flatten())
        if clfDec_linear_layers is not None:
            clfDec_linear_layers[0]["in_features"] = featlen
            for layer in clfDec_linear_layers:
                self.clfDec.extend(nn.Sequential(nn.Linear(**layer),
                                                     nn.ReLU(),
                                                     nn.Dropout(clf_dropout)))
                featlen = clfDec_linear_layers[-1]["out_features"]
        ##### CLASSIFIER #####
        self.clfClf = nn.Sequential(nn.Linear(featlen, n_classes),
                                    nn.Sigmoid() if clf_activation=='sigmoid' else nn.Softmax(dim=1) if clf_activation=='softmax' \
                                        else nn.LogSigmoid() if clf_activation=='logsigmoid' else nn.LogSoftmax(dim=1) if clf_activation=='logsoftmax' \
                                        else nn.Identity(dim=1),
                                    )
    def forward(self, x: Tensor) -> Tensor:
        """
        Classifier Head

        Parameters
        ----------
        x: Tensor
            shape (B x 1 x N x t)

        Returns
        -------
        Tensor
            <**pred**>: Tensor
                shape: (B x n_classes)

        """
        assert [*x.shape[1:]] == [1, self.N, self.outlen]
        return self.clfClf(self.clfDec(x))
        ########################################
    def _conv2d_block(self, layers: Sequence[Dict], batchnorm:bool, dropout: float):
        block = nn.Sequential()
        for layer in layers:
            if batchnorm:
                block.extend(nn.Sequential(
                    nn.Conv2d(**layer),
                    nn.BatchNorm2d(layer["out_channels"]),
                    nn.ReLU(),
                    nn.Dropout2d(dropout),
                ))
            else:
                block.extend(nn.Sequential(
                    nn.Conv2d(**layer),
                    nn.ReLU(),
                    nn.Dropout2d(dropout),
                ))
        return block

# %% MAIN MODELS
class EEGenoNet(nn.Module):
    """
    [EEGenoNet]

    Forward
    ----------
    **Inputs**:
        ``x``: Tensor
            time-series EEG. shape: `B x F x N x T`

        ``get_token``: bool
            if True, return separated <**tokens**> (default False)
                - temporal (intra-channel) cls tokens: `d x N`
                - spatial (inter-channel) cls tokens: `d x T'`
                - spectral-channel tokens: `F x d`

    **Outputs**:
        if ``self.apply_clf_head``:
            * if ``get_token``: <**(tokens)**, **pred**>: Tuple[Tensor, Tensor, Tensor], Tensor
            * else: <**pred**>: Tensor
        else:
            * if ``get_token``: <**(tokens)**>: Tensor, Tensor, Tensor
            * else: <**spec_token**, **x**>: Tensor(`B x F x d`), Tensor (`B x d x (1+N) x (1+T')`)


    Parameters
    ----------
    args: edict

            - 'apply_clf_head'
            - 'F', 'N', 'T' (input data shape)
            - 'embdim', 'k' (kernel for dyConv)
            - 'n_classes'
            - 'patch_conv_layers' ([{"in_channels": ..., }, ...])
            - 'intp_mode': Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-exact']
            - 'dot_conv_outdims': Sequence[int]
            - 'nhead' (for MHA), 'nlayer' (# of former blocks), 'enc_dropout'
            - 'dconv_bias', 'dconv_dropout'
            - 'clfDec_conv_layers', 'clfDec_linear_layers'(Sequence[Dict]|None),
            - 'clf_dropout', 'clf_activation' (Literal['sigmoid', 'softmax', 'logsoftmax', 'logsoftmax', 'logsoftmax', 'none']), 'clf_batchnorm': bool

    Attributes
    ----------
    outlen: int
        # of time-axis tokens (=T')
    Encoder: nn.Module
        patchEmbed + specTokenEmbed+ Intp + biaxialEmbed + formerBlocks
    dyConv: nn.Module
        specTempFGB – Conv2d (d, d, (F, 1); ReLU) + Conv2d (d, d, (1, 1); ReLU)

        specSpatFGB – Conv2d (d, d, (F, 1); ReLU) + Conv2d (d, d, (1, 5), stride=(1, 2); ReLU) + AdaptiveAvgPool2d((1, k))

        tempSpatFGB – Conv2d (d, d, (1, 1); Sigmoid) + temp_spat_token (itself)

        unfold
    Classifier: nn.Module
        clfDec + clfClf

    """
    def __init__(self, args:edict):
        super().__init__()
        self.apply_clf_head = args.apply_clf_head

        ##### MODULES #####
        self.Encoder = Encoder(args.F, N=args.N, T=args.T, embdim=args.embdim,
                               patch_conv_layers=args.patch_conv_layers, intp_mode=args.intp_mode,
                               dot_conv_outdims=args.dot_conv_outdims, nlayer=args.nlayer, nhead=args.nhead,
                               enc_dropout=args.enc_dropout)
        self.outlen = self.Encoder.outlen
        if args.apply_clf_head:
            self.dyConv = DyConv(F=args.F, N=args.N, T=args.T, outlen=self.outlen,
                                 embdim=args.embdim, k=args.k, dconv_bias=args.dconv_bias, dconv_dropout=args.dconv_dropout)
            self.Classifier = ClassifierHead(N=args.N, outlen=self.outlen, n_classes=args.n_classes,
                                             clfDec_conv_layers=args.clfDec_conv_layers,
                                             clfDec_linear_layers=args.clfDec_linear_layers,
                                             clf_dropout=args.clf_dropout, clf_batchnorm=args.clf_batchnorm, clf_activation=args.clf_activation)
    def forward(self, x:Tensor,
                get_token: bool=False) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tuple[Tensor, Tensor, Tensor], Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Run EEGenoNet forward pass.

        Parameters
        ----------
        x: Tensor
            time-series EEG.

            shape: `B x F x N x T`
        get_token: bool
            if True, return separated <**tokens**> (default False)

            - temporal (intra-channel) cls tokens: `d x N`
            - spatial (inter-channel) cls tokens: `d x T'`
            - spectral-channel tokens: `F x d`

        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tuple[Tensor, Tensor, Tensor], Tensor], Tuple[Tensor, Tensor, Tensor]]
            if 'self.apply_clf_head':

                * if 'get_token': <**(tokens)**, **pred**>: Tuple[Tensor, Tensor, Tensor], Tensor
                * else: <**pred**>: Tensor
            else:

                * if 'get_token': <**(tokens)**>: Tensor, Tensor, Tensor
                * else: <**x_sc**, **x**>: Tensor(`B x F x d`), Tensor (`B x d x (1+N) x (1+T')`)

        """
        spec_token, x = self.Encoder(x)     # spec_token (B x F x d) # x (B x d x (1+N) x (1+t))
        if get_token:
            temp_token, spat_token = x[:, :, 1:, 0], x[:, :, 0, 1:]
        if self.apply_clf_head:
            x = self.Classifier(self.dyConv(spec_token, x))     # pred (B x n_classes)
            if get_token:
                return (temp_token, spat_token, spec_token), x
            else: return x
        # -----
        if get_token:
            return temp_token, spat_token, spec_token
        else:
            return spec_token, x


class EEGAutoEncoder(nn.Module):
    """
    EEGenoNet Encoder + Reconstruction Decoder

    Forward
    ----------
    **Input:**
        <**x**>: Tensor
            EA aligned input time-series EEG signals. shape (B x F x N x T)

    **Output:**
        <**x**>: Tensor
            reconstructed EEG signals. shape (B x F x N x T)


    Parameters
    ----------
    args: edict

            - 'apply_clf_head': must be False
            - 'ae_activation': Literal['gelutanh', 'gelu', 'tanh', 'sigmoid']
            - 'F', 'N', 'T' (input data shape)
            - 'embdim'
            - 'patch_conv_layers' ([{"in_channels": ..., }, ...])
            - 'intp_mode': Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-exact']
            - 'dot_conv_outdims': Sequence[int]
            - 'nhead' (for MHA), 'nlayer' (# of former blocks), 'enc_dropout'


    Attributes
    ----------
    Encoder: nn.Module
        patchEmbed + specTokenEmbed+ Intp + biaxialEmbed + formerBlocks
    Decoder: nn.ModuleList[nn.Sequential]
        F-band wise convolutional blocks.

        Each block:

            ConvTranspose2d(1, args.embdim,
                            kernel_size=self.conv_kernels[-1],
                            stride=self.conv_strides[-1],
                            padding=self.conv_paddings[-1]))

            GELU

            Conv2d(args.embdim, 1, kernel, stride, padding)

            GELUTanh() | nn.Sigmoid() | nn.Tanh() | nn.GELU()
    """
    def __init__(self, args:edict):
        super().__init__()
        assert args.apply_clf_head==False
        assert 'ae_activation' in args

        self.conv_kernels = [layer[0]["kernel_size"] for layer in args.patch_conv_layers]       # (1, 100), (1, 50), (1, 25)
        self.conv_paddings = [layer[0]["padding"] for layer in args.patch_conv_layers]          # (0, 40), (0, 20), (0, 10)
        self.conv_strides = [layer[0]["stride"] for layer in args.patch_conv_layers]            # (1, 40), (1, 20), (1, 10)
        kernel_length_ratio = [stride[1]//self.conv_strides[-1][1] for stride in self.conv_strides]     # (4, 2, 1)
        ##### MODULES #####
        self.Encoder = EEGenoNet(args)
        self.Decoder = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(1, args.embdim,
                                             kernel_size=kernel,
                                             stride=stride,
                                             padding=padding),
                          nn.SELU(),

                          nn.Conv2d(args.embdim, 1,
                                    kernel_size=kernel,
                                    stride=(1, kr),
                                    padding=(0, 0)),

                          GELUTanh() if args.ae_activation == 'gelutanh' \
                                else nn.Sigmoid() if args.ae_activation == 'sigmoid' \
                                else nn.Tanh() if args.ae_activation == 'tanh' \
                                else nn.GELU if args.ae_activation == 'gelu' \
                                else nn.Identity(),
                          )
            for kernel, stride, padding, kr in zip(self.conv_kernels, self.conv_strides, self.conv_paddings, kernel_length_ratio)
        ])
        self.outLayer = nn.Sequential(
            Interpolation(size=(args.N, args.T), mode=args.intp_mode)
        )

    def forward(self, x:Tensor) -> Tensor:
        """

        Parameters
        ----------
        x: Tensor
            EA aligned input time-series EEG signals. shape (B x F x N x T)

        Returns
        -------
        Tensor
            ``x_reconst``: reconstructed EEG signals. shape (B x F x N x T)

        """
        temp_token, spat_token, spec_token = self.Encoder(x, get_token=True)
        x = torch.einsum('bfd, bdn, bdt -> bfnt', spec_token, temp_token, spat_token)

        x = torch.cat([self.outLayer(block(x[:, [f_i]]))
                       for f_i, block in enumerate(self.Decoder)],
                      dim=1)        # x (B x F x N x T)
        return x

