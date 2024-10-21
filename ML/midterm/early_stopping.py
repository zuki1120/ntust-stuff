import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience = 3, lr_patience = 2, lr_factor = 0.2, verbose = True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.lr_counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor

    def __call__(self, val_loss, model, optimizer):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            self.lr_counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.lr_counter >= self.lr_patience:
                self.reduce_lr(optimizer)  # 降低學習率
                self.lr_counter = 0  # 重置 lr_counter
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.lr_counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

    def reduce_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * self.lr_factor
            param_group['lr'] = new_lr
        if self.verbose:
            print(f'Reducing learning rate to {new_lr:.6f}')