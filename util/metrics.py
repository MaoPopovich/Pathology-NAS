import torch
from typing import List
import random
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import os

@torch.no_grad()
def iou_score(x: torch.Tensor, y: torch.Tensor, eval_class_idx: List[int]):
    smooth = 1e-8
    max_val, _ = torch.max(x, dim=1, keepdim=True)
    x = (x == max_val)[:, eval_class_idx]
    y = y.bool()[:, eval_class_idx]
    intersection = torch.logical_and(x, y)
    cardinality = torch.logical_or(x, y)

    intersection = torch.sum(intersection.long(), dim=(0, 2, 3))
    cardinality = torch.sum(cardinality.long(), dim=(0, 2, 3))

    iou = intersection / (cardinality + smooth)
    iou = torch.mean(iou)
    return iou, intersection, cardinality


@torch.no_grad()
def dice_score(x: torch.Tensor, y: torch.Tensor, eval_class_idx: List[int]):
    smooth = 1e-8
    max_val, _ = torch.max(x, dim=1, keepdim=True)
    x = (x == max_val)[:, eval_class_idx]
    y = y.bool()[:, eval_class_idx]
    intersection = torch.logical_and(x, y).long() * 2
    cardinality = x.long() + y.long()

    intersection = torch.sum(intersection, dim=(0, 2, 3))
    cardinality = torch.sum(cardinality, dim=(0, 2, 3))

    dice = intersection / (cardinality + smooth)
    dice = torch.mean(dice)
    return dice, intersection, cardinality

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if target.dim() == 3:
        print("ok")
        pool = nn.AdaptiveAvgPool2d(1)
        target = pool(target.float()).squeeze().long()
    
    elif target.dim() == 2:
        target = torch.argmax(target, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    output = torch.randn(128,200)
    # output = output.softmax(dim=-1)
    target = torch.randint(low=0,high=200,size=(128,))
    prec1, = accuracy(output, target)
    print(prec1)