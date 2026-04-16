import os
import numpy as np
import pandas as pd
import torch
torch.manual_seed(42)
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import seaborn as sns
from importlib import resources
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassAUROC
from torchmetrics.classification import MulticlassAccuracy

config_dir = f'{resources.files("hlarchical")}/config'
data_dir = f'{resources.files("hlarchical")}/data'

class CustomLoss(torch.nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.ground_truth_gated_loss = False
        self.gated_loss_lambda = 1.0
        self.maps = {}
        if cfg:
            if hasattr(cfg, 'maps_file') and os.path.exists(cfg.maps_file):
                df = pd.read_table(cfg.maps_file, header=0, sep='\t')
                for n in range(df.shape[0]):
                    head = df['head'].iloc[n]
                    head_idx = df['head_idx'].iloc[n]
                    parent = df['parent'].iloc[n]
                    parent_val = df['parent_val'].iloc[n]
                    self.maps[head] = [head_idx, parent, parent_val]

            if hasattr(cfg, 'ground_truth_gated_loss') and cfg.ground_truth_gated_loss:
                self.ground_truth_gated_loss = True
                if hasattr(cfg, 'gated_loss_lambda'):
                    self.gated_loss_lambda = cfg.gated_loss_lambda 
                print(f'Using ground truth gated loss with lambda {self.gated_loss_lambda}')

    def forward(self, y_pred, y_true):
        loss = torch.tensor(0.0, device=y_true.device)
        if not self.ground_truth_gated_loss:
            for head in y_pred:
                y_p = y_pred[head]
                head_idx = self.maps[head][0]
                y_t = y_true[:, :, head_idx]
                ls = self.loss_fn(y_p, y_t)
                loss += ls
        else:
            loss1 = torch.tensor(0.0, device=y_true.device)
            loss2 = torch.tensor(0.0, device=y_true.device)
            for head in y_pred:
                y_p = y_pred[head]
                head_idx = self.maps[head][0]
                y_t = y_true[:, :, head_idx]
                parent = self.maps[head][1]
                parent_idx = self.maps[parent][0]
                parent_val = self.maps[head][2]
                if parent == '.':
                    ls = self.loss_fn(y_p, y_t)
                    loss1 += ls
                else:
                    for i in range(y_t.shape[0]):
                        for j in range(y_t.shape[1]):
                            if y_true[i, j, parent_idx] == parent_val:
                                ls = self.loss_fn(y_p[i, :, j].unsqueeze(0), y_t[i, j].unsqueeze(0))
                                loss2 += ls
            loss = loss1 + loss2/y_true.shape[0] * self.gated_loss_lambda
        return loss

class Config:
    def __init__(self, config):
        for k, v in config.items():
            if k in ['learning_rate', 'weight_decay']:
                v = float(v)
            elif k in ['NWD']:
                v = eval(v)
            setattr(self, k, v)

class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.val_loss_min = np.inf
        self.counter = 0
        self.stopped = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss
            self.counter = 0
            self.best_epoch = epoch
        elif val_loss > self.val_loss_min + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True


class CustomAccuracy(MulticlassAccuracy):
    def __init__(self, num_classes, per_allele=True, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        # 'sum' tells TorchMetrics to add values across GPUs in distributed training
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.per_allele = per_allele

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape

        if self.per_allele:
            preds = preds.view(-1)
            target = target.view(-1)
            mask = target != 0
            preds = preds[mask]
            target = target[mask]
        else:
            mask = target.sum(dim=1) != 0
            preds = preds[mask]
            target = target[mask]

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        if self.total == 0:
            accuracy = torch.tensor(0.0)
        else:
            accuracy = self.correct.float() / self.total
        res = [accuracy, self.correct, self.total]
        return [float(res[0]), int(res[1]), int(res[2])]

def accuracy_avg_digit(accuracy, digits, average='micro'):
    A = {}
    for d in digits:
        L = []
        for a in digits[d]: 
            L.append(accuracy[a])
        mat = np.array(L)
        if average == 'micro':
            correct =  np.sum(mat[:, 1])
            totoal = np.sum(mat[:, 2])
            acc = correct / totoal if totoal > 0 else 0.0
        elif average == 'macro':
            acc = np.mean(mat[:, 0])
        A[d] = acc
    return A
