import torch
import torch.nn.functional as F
import torch.nn as nn


def DiceLoss(x: torch.Tensor, y: torch.Tensor):
    x = torch.softmax(x, dim=1)
    # print(x.shape, y.shape)
    smooth = 1e-8
    intersection = 2 * x * y
    cardinality = x + y
    intersection = torch.sum(intersection.float(), dim=(0, 2, 3))
    cardinality = torch.sum(cardinality.float(), dim=(0, 2, 3))
    dice = intersection / (cardinality + smooth)
    dice = torch.mean(dice)
    return 1 - dice, intersection, cardinality


def CrossEntropyAvoidNaN(x: torch.Tensor, y: torch.Tensor, label_smoothing: float):
    """ Each pixel of image target is an indicator label
    """
    if y.ndim == 3:
        bs, h, w = y.size()
        pool = nn.AdaptiveAvgPool2d(1)
        y_t = y.float()
        y_t = pool(y_t).squeeze()
        y_t = y_t.long()
    else:
        y_t = y.long()
    # print(y_t.shape)
    # ce_loss = F.cross_entropy(x, y, reduction="none", label_smoothing=label_smoothing)
    # ce_loss = (((ce_loss / h).sum(1) / w).sum(1) / bs).sum()
    ce_loss = F.cross_entropy(x, y_t, label_smoothing=label_smoothing)
    return ce_loss
