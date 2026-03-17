"""IMPORT PACKAGES"""
import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np

""""""""""""""""""""""""""""""""""""""""""""""""
"""" DEFINE HELPER FUNCTIONS FOR LOSS FUNCTION"""
""""""""""""""""""""""""""""""""""""""""""""""""


def construct_loss_function_cls(opt):

    # Define possible choices for classification loss
    if opt.cls_criterion == 'BCE':
        cls_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([opt.cls_criterion_weight], dtype=torch.float32))
    elif opt.cls_criterion == 'CE':
        cls_criterion = nn.CrossEntropyLoss()
    elif opt.cls_criterion == 'Focal':
        cls_criterion = FocalLoss_Cls(smooth=1e-6, alpha=opt.focal_alpha_cls, gamma=opt.focal_gamma_cls)
    else:
        raise Exception('Unexpected Classification Loss {}'.format(opt.cls_criterion))

    return cls_criterion

# Custom Focal Loss for classification
class FocalLoss_Cls(nn.Module):
    def __init__(self, alpha, gamma, smooth=1e-6):
        super(FocalLoss_Cls, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def __call__(self, preds, target):

        # Check whether the batch sizes of prediction and target match [BS, c, h, w]
        assert preds.shape[0] == target.shape[0], "pred & target batch size don't match"

        # Compute predictions after sigmoid activation
        preds = self.sigmoid(preds)

        # Flatten the prediction and target. Shape = [BS, c*h*w]]
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute Binary Cross Entropy
        BCE = F.binary_cross_entropy(preds, target, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1. - BCE_EXP) ** self.gamma * BCE

        return focal_loss


""""""""""""""""""""""""""""""""""""""""""
"""" DEFINE HELPER FUNCTIONS FOR OPTIMIZER"""
""""""""""""""""""""""""""""""""""""""""""


def construct_optimizer(optim, parameters, lr):

    # Define possible choices
    if optim == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999),
                                     eps=1e-07, amsgrad=True, weight_decay=1e-4)
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9)
    else:
        raise Exception('Unexpected Optimizer {}'.format(optim))

    return optimizer


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""" DEFINE HELPER FUNCTIONS FOR LEARNING RATE SCHEDULER"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def construct_scheduler(schedule, optimizer, lr, metric="val_loss"):

    # Define possible choices
    if schedule == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                                               factor=0.1, patience=5, min_lr=lr/1000)

        return {"scheduler": scheduler,
                "monitor": metric,
                "interval": "epoch"}

    elif schedule == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

        return {"scheduler": scheduler,
                "interval": "epoch"}

    elif schedule == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=2, T_mult=2,
                                                                         eta_min=lr/1000, last_epoch=-1)

        return {"scheduler": scheduler,
                "interval": "epoch"}

    else:
        return None
