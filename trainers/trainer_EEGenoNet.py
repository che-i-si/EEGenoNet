import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import math, os

# %% TRAINER
class Trainer(object):
    def __init__(self,
                 model,
                 args,
                 device,
                 save_path= "checkpoints/pretrained.pt",     ):
        """

        Parameters
        ----------
        model: backbones.EEGenoNet.EEGenoNet
            Model to be trained.
        args: easydict.EasyDict
        device: str|torch.device
        save_path: str|None
        """
        self.model = model.to(device)

        self.epochs = args.clf_epochs
        self.lr = args.clf_lr
        self.device = device
        self.n_classes = args.n_classes
        self.save_path = save_path

        ##### CRITERION #####
        self.criterion = torch.nn.CrossEntropyLoss()
        ##### OPTIMIZER #####
        self.optim = torch.optim.Adam(model.parameters(), lr=args.clf_lr, weight_decay=args.clf_weight_decay)
        ##### FOR EARLY-STOPPING
        self.early_stopping_patience = args.clf_early_stopping_patience if 'clf_early_stopping_patience' in args else 10
        self.min_run_epochs = args.clf_min_run_epochs if 'clf_min_run_epochs' in args else 30
        self.patience = 0
        self.best_loss = 10000

    def _init_running(self):
        self.next = True
        self.val_loss, self.train_loss = None, None
        self.current_epoch = 0
        self.patience = 0
        self.best_loss = 10000

    ##################################################################################
    def training_step(self, batch):
        self.optim.zero_grad()
        ### RUN
        pred = self.model(*[x.to(self.device) for x in batch[:-1]])
        if self.n_classes == 1: pred = pred.flatten()
        true = batch[-1] if self.n_classes == 1 \
            else torch.as_tensor(batch[-1], dtype=torch.int64)
        ### UPDATE
        self.optim.zero_grad()
        loss = self.criterion(pred.cpu(), true)
        loss.backward(retain_graph=True)
        self.optim.step()
        # pred = pred.argmax(dim=1)

        return loss.item()

    def validation_step(self, batch, return_pred=False, pred_argmax=True):
        self.model.eval()
        ### RUN
        pred = self.model(*[x.to(self.device) for x in batch[:-1]])
        if self.n_classes == 1: pred = pred.flatten()
        true = batch[-1] if self.n_classes == 1 \
            else torch.as_tensor(batch[-1], dtype=torch.int64)
        loss = self.criterion(pred.cpu(), true)
        if pred_argmax and (self.n_classes>1): pred = pred.argmax(dim=1)

        if return_pred:
            return loss.item(), pred.detach().cpu()
        return loss.item()


    ##################################################################################
    def train_epoch(self, train_loader: torch.utils.data.DataLoader):
        self._adjust_learning_rate()
        self.model.train()

        losses = []
        for batch_idx, batch in enumerate(train_loader):
            # ===== Train
            loss = self.training_step(batch)
            losses.append(loss)
        self.train_loss = sum(losses)/len(losses)


    def validation_epoch_end(self, val_loader: torch.utils.data.DataLoader):
        self.model.eval()

        losses = []
        for batch in val_loader:
            loss = self.validation_step(batch, return_pred=False)
            losses.append(loss)
        self.val_loss = sum(losses) / len(losses)
        self.next = self._early_stopping()

        return self.val_loss

    ##################################################################################
    def fit(self, train_loader, val_loader, save=True, init_run=None):
        """
        Train and validate the model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            training data samples
        val_loader: torch.utils.data.DataLoader|None
            validation samples. Used for early stopping.
        save: bool
            save state_dict of the trained model (default: True)
        init_run: bool|None
            re-initialize training settings (e.g., best_loss, current_epoch, ...)
        """
        if init_run: self._init_running()
        # -----
        progress_bar = tqdm(range(self.epochs), colour='WHITE', ncols=150, leave=True)
        for epoch in progress_bar:
            self.current_epoch = epoch
            self.train_epoch(train_loader)
            # ===== Validation
            if not val_loader is None:
                self.validation_epoch_end(val_loader)
                if self.next == False:
                    progress_bar.close()
                    print(f'\nEarly stopped at epoch {self.current_epoch}\tVal Loss: {self.val_loss:.4f}')
                    break
                else:
                    progress_bar.set_postfix({'train_loss': self.train_loss,
                                              'val_loss': self.val_loss})
            else:
                progress_bar.set_postfix({'train_loss': self.train_loss,})

        # ===== END OF EPOCHS
        if save:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.save_path)


    def test(self,
             test_loader,
             verbose=True,
             return_pred=False,
             pred_argmax=True):
        """
        Evaluate the model.

        Parameters
        ----------
        test_loader: torch.utils.data.DataLoader
        verbose: bool
            if True, print test loss and mean predicted value
        return_pred: bool
        pred_argmax: bool

        Returns
        ----------
        float|(float, float)|(float, float, np.ndarray)|(float, np.ndarray)
            ``test_loss``: float
                Cross-entropy loss
            ``test_acc``: float
                when ``self.n_classes`` > 1
            ``test_pred``: np.ndarray
                when ``return_pred`` is True

        """
        self.model.eval()
        test_loss = []
        test_pred, test_true = [], []
        # ===== Test
        if self.n_classes==1: pred_argmax = False
        for batch in test_loader:
            loss, pred = self.validation_step(batch, return_pred=True, pred_argmax=pred_argmax)
            test_loss.append(loss)
            test_pred.append(pred.detach().cpu().numpy())
            test_true.append(batch[-1].numpy())
        # ======
        test_loss = sum(test_loss) / len(test_loss)
        test_pred, test_true = np.concatenate(test_pred), np.concatenate(test_true)
        if self.n_classes>1:
            test_acc = accuracy_score(test_true,
                                      test_pred if test_pred.ndim==1 else np.argmax(test_pred, axis=1))
            # ===== PRINT
            if verbose:
                print('------------------------------------------------------------\n')
                print(f'Test Loss: {test_loss:.4f}\nTest Acc: {test_acc:.4f}')
                print('\n------------------------------------------------------------')
            if return_pred: return test_loss, test_acc, test_pred
            else: return test_loss, test_acc
        else:
            if verbose:
                print('------------------------------------------------------------\n')
                print(f'Test Loss: {test_loss:.4f}')
                print('\n------------------------------------------------------------')
            if return_pred: return test_loss, test_pred
            else: return test_loss

    ##################################################################################
    def _adjust_learning_rate(self):
        """Decay the learning rate based on schedule"""
        warmup_epoch = self.epochs // 10 if self.epochs <= 100 else 40

        if self.current_epoch < warmup_epoch:
            cur_lr = self.lr * self.current_epoch / warmup_epoch + 1e-9
        else:
            cur_lr = (
                self.lr* 0.5
                * (1.0
                    + math.cos(
                        math.pi
                        * (self.current_epoch - warmup_epoch)
                        / (self.epochs - warmup_epoch)
                    )
                )
            )
        # ----- 적용
        for param_group in self.optim.param_groups:
            param_group["lr"] = cur_lr

    def _early_stopping(self):
        if self.current_epoch >= self.min_run_epochs:
            if self.val_loss < self.best_loss:
                self.best_loss = self.val_loss
                self.patience = 0
                return True
            else:
                self.patience += 1
                if self.patience >= self.early_stopping_patience:
                    # print(f'\nEarly stopped at epoch {self.current_epoch}\tVal Loss: {self.val_loss:.4f}')
                    return False
                return True
        else:
            return True

    ##################################################################################

