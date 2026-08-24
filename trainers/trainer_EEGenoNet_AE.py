import torch
from tqdm import tqdm
import math, os
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self,
                 model,
                 args,
                 device,
                 save_path="checkpoints/pretrained.pt", ):
        """
        Trainer for EEGenoNet Autoencoder (pre-training).

        Parameters
        ----------
        model: backbones.EEGenoNet.EEGAutoEncoder
        args: easydict.EasyDict
        device: str|torch.device
        save_path: str
        """
        self.model = model.to(device)
        ##### FOR TRAINING
        self.epochs = args.ae_epochs
        self.lr = args.ae_lr
        self.device = device
        #####
        self.criterion = torch.nn.L1Loss()
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=args.ae_weight_decay,
            betas=(0.9, 0.95),
        )
        ##### FOR SAVE MODEL
        self.save_path = args.model_savedir + save_path
        ##### FOR EARLY-STOPPING
        self.early_stopping_patience = args.early_stopping_patience if 'ae_early_stopping_patience' in args else 10
        self.min_run_epochs = args.min_run_epochs if 'ae_min_run_epochs' in args else 30
        self.patience = 0
        self.best_loss = 10000

    ##################################################################################
    def training_step(self, batch):
        self.optim.zero_grad()
        ### RUN
        pred = self.model(batch[0].to(self.device))
        pred = pred.to('cpu')
        ### UPDATE
        self.optim.zero_grad()
        loss = self.criterion(pred.cpu(), batch[0].cpu())
        loss.backward()
        self.optim.step()
        del pred

        return loss.item()

    def validation_step(self, batch, get_pred:bool=False):
        self.model.eval()
        ### RUN
        pred = self.model(batch[0].to(self.device))
        pred = pred.detach().cpu()
        loss = self.criterion(pred.cpu(), batch[0].cpu()).item()
        if get_pred: return pred, loss
        return loss


    ##################################################################################
    def train_epoch(self, train_loader: torch.utils.data.DataLoader):
        self._adjust_learning_rate()
        self.model.train()

        tr_loss = []
        for batch_idx, batch in enumerate(train_loader):
            # ===== Train
            loss = self.training_step(batch)
            tr_loss.append(loss)
        tr_loss = sum(tr_loss) / len(tr_loss)
        self.train_loss = tr_loss


    def validation_epoch_end(self, val_loader: torch.utils.data.DataLoader, plot_trained:bool=False):
        self.model.eval()
        val_loss = []
        for i, batch in enumerate(val_loader):
            if (plot_trained) and (i == len(val_loader) - 1) and (self.current_epoch % 5 == 0):
                chan_idx = 12
                pred, loss = self.validation_step(batch, get_pred=True)
                pred, true = pred[0, :, chan_idx].numpy(), batch[0][0, :, chan_idx].numpy()
                # -----
                fig, axes = plt.subplots(pred.shape[0], 2, figsize=(8*2, 2*pred.shape[0]))
                for i, ax in enumerate(axes.flatten()):
                    row, col = i // 2, i % 2
                    if col == 0:
                        ax.plot(true[row], color='steelblue', lw=0.5)
                    else:
                        ax.plot(pred[row], color='steelblue', lw=0.5)
                fig.suptitle(f'Epoch: {self.current_epoch}, Chan idx: {chan_idx}', fontsize=10)
                fig.tight_layout()
                plt.show()
            else:
                loss = self.validation_step(batch, get_pred=False)
            val_loss.append(loss)
        val_loss = sum(val_loss) / len(val_loss)
        self.val_loss = val_loss
        self.next = self._early_stopping()

        return self.val_loss

    ##################################################################################
    def fit(self, train_loader, val_loader, save=True, plot_trained=False):
        """
        Train and validate the model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            training data samples
        val_loader: torch.utils.data.DataLoader
            validation samples. Used for early stopping
        save: bool
            save state_dict of the trained model (default: True)
        plot_trained: bool
            For the first validation sample, the signal from the 13th channel of the
            generated sample was plotted every 5 epochs (default: False)

        """
        progress_bar = tqdm(range(self.epochs), colour='WHITE', ncols=150, leave=True)
        for epoch in progress_bar:
            self.current_epoch = epoch
            self.train_epoch(train_loader)      # self.val_loss
            # ===== Validation
            self.validation_epoch_end(val_loader, plot_trained=plot_trained)
            progress_bar.set_postfix({"tr_loss": self.train_loss, "val_loss": self.val_loss})
            if self.next == False:
                break

        # ===== END OF EPOCHS
        if save:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.save_path)

    def test(self, test_loader, verbose=True):
        """
        Evaluate the model.

        Parameters
        ----------
        test_loader: torch.utils.data.DataLoader
        verbose: bool
            if True, print test loss and mean predicted value

        Return
        ----------
        float
            ``test_loss``: (float) L1-loss

        """
        test_loss = []
        # ===== Test
        for batch in test_loader:
            loss = self.validation_step(batch)
            test_loss.append(loss)
        test_loss = sum(test_loss) / len(test_loss)
        # ===== PRINT
        if verbose:
            print('------------------------------------------------------------\n')
            print(f'Test Loss: {test_loss}')
            print('\n------------------------------------------------------------')
        return test_loss

    ##################################################################################

    def _adjust_learning_rate(self):
        cur_lr = (
            self.lr
            * 0.5
            * (1.0 + math.cos(math.pi * self.current_epoch / self.epochs))
        )

        for param_group in self.optim.param_groups:
            if "fix_lr" in param_group and param_group["fix_lr"]:
                param_group["lr"] = self.lr
            else:
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
                    print(f'\nEarly stopped at epoch {self.current_epoch}\tVal Loss: {self.val_loss:.03f}\n')
                    return False
                return True
        else:
            return True

    ##################################################################################

