import os
import json
import math
import time
from thop import profile
import numpy as np
from decimal import Decimal
import argparse
import torch.distributed as dist
import torch.optim as optim
import builtins
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from typing import Tuple, List
import backbone as archs
from util.cost_func import DiceLoss, CrossEntropyAvoidNaN
from util.metrics import iou_score, dice_score, accuracy
from util.meter import AverageMeter, ProgressMeter, InstantMeter
from util.torch_dist_sum import torch_dist_sum
from util.dist_init import random_choice, set_seed
from util.samplers import RASampler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from data.dataset import init_imagenet, BCSSDataset, CancerInst, build_transform, build_mask_transform
from torch.distributed import init_process_group, destroy_process_group
from data.augmentation import TranAugment, val_aug
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import segmentation_models_pytorch as smp
parser = argparse.ArgumentParser()

# Supernet Setting
parser.add_argument("--layers", type=int, default=20, help="number of choice_block layers in CNNs or operations in ViT")
parser.add_argument("--num_choices", type=int, default=4, help="number of choices per layer in CNNs or per op in ViT")
parser.add_argument("--use_lmdb", action='store_true', help="whether or not use LMDB dataset")

# model settings
parser.add_argument('--model4task', type=str, default='unet4cls', required=True)
parser.add_argument('--arch', type=str, default="ResNet")
parser.add_argument("--model", type=str, default="vit")
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--eval_classes", type=int, default=3)
parser.add_argument("--dataset", type=str, default="bcss")
parser.add_argument("--ft_epochs", type=int, default=10)

# training settings
parser.add_argument("--seed", type=int, default=42, help="training seed")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--alpha", type=float, default=0.0)
parser.add_argument("--split", type=int, default=0)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--train_full", default=False, action="store_true")
parser.add_argument("--eval_only", default=False, action="store_true")
parser.add_argument("--freeze_encoder", default=False, action="store_true")
parser.add_argument("--w_rcc", default=False, action="store_true")
parser.add_argument("--finetune_ckpt", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument(
    "--pretrained_weights",
    type=str,
    default=None,
    # choices=[None, "IMAGENET1K_V1", "IMAGENET1K_V2"],
)

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + \
                            "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

parser.add_argument('--repeated-aug', action='store_true')

# * Random Erase params
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')


args = parser.parse_args()
set_seed(args.seed)
unet_map = {
    'bcss':{10: [7, 3, 5, 5, 3, 7, 3, 7], 20: [3, 5, 3, 7, 3, 5, 7, 3], 30: [3, 3, 5, 5, 7, 7, 3, 3], 40: [3, 3, 3, 5, 7, 3, 5, 3]},
    'cancerinst':{10: [5, 7, 3, 5, 7, 3, 7, 5], 20: [3, 5, 5, 3, 7, 3, 7, 5], 30: [3, 3, 5, 5, 7, 7, 3, 3], 40: [3, 3, 5, 5, 7, 7, 3, 3]}
}

if args.model == 'cnn':
    candidate = unet_map[args.dataset][args.ft_epochs]

@torch.no_grad()
def batch_process_label(label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    class0 = label == 0
    class1 = label == 1
    if args.dataset == 'bcss':
        class2 = label == 2
        stack_label = torch.stack([class0, class1, class2], dim=1)   # label for ce_loss, stack_label for dice loss
    else:
        stack_label = torch.stack([class0, class1], dim=1)
    stack_label = (stack_label > 0.5).float().squeeze()
    # print(stack_label.shape)
    return label.long(), stack_label

def adjust_learning_rate(
    optimizer: torch.optim.Optimizer, epoch: int, i: int, iteration_per_epoch: int
) -> None:
    warm_up = max(1, args.epochs // 20)
    base_lr = args.lr
    epochs = args.epochs
    if epoch < warm_up:
        T = epoch * iteration_per_epoch + i
        warmup_iters = warm_up * iteration_per_epoch
        lr = base_lr * T / warmup_iters
    else:
        min_lr = base_lr / 1000
        T = epoch - warm_up
        total_iters = epochs - warm_up
        lr = (
            0.5 * (1 + math.cos(1.0 * T / total_iters * math.pi)) * (base_lr - min_lr)
            + min_lr
        )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def train(
    train_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    select_id : List[int],
) -> Tuple[float, float]:
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    curr_lr = InstantMeter("LR", ":6.5f")
    ce_losses = AverageMeter("CE", ":.4e")
    dice_losses = AverageMeter("DICE", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, curr_lr, ce_losses, dice_losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    if args.freeze_encoder:
        if hasattr(model, "module"):
            model.module.encoder.eval()
        else:
            model.encoder.eval()

    iteration_per_epoch = len(train_loader)

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # inputs_len = len(inputs)  # data augmentation to generate two views
        # for img_idx in range(inputs_len):
        #     inputs[img_idx] = inputs[img_idx].cuda(non_blocking=True)
        bs = input.size(0)
        # inputs = torch.cat(inputs)
        if target.shape[1] > 1:
            target = target[:, -1, :, :]

        adjust_learning_rate(optimizer, epoch, i, iteration_per_epoch)
        ce_target, dice_target = batch_process_label(target)  # for ce_loss and dice_loss
        data_time.update(time.time() - end)

        loss = 0
        with torch.cuda.amp.autocast(enabled=True):
            if args.model == 'cnn':
                choice = select_id
                output = model(input, choice)
            else:
                depth_idx, attn_idx, mlp_idx = select_id['depth'], select_id['heads'], select_id['mlp_ratios']
                output = model(input, depth_idx, attn_idx, mlp_idx)
            # print(output.shape, dice_target.shape)
            dice_loss, _, _ = DiceLoss(output, dice_target)
            loss += dice_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        curr_lr.update(optimizer.param_groups[0]["lr"])
        ce_losses.update(0, bs)
        dice_losses.update(dice_loss.item(), bs)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            progress.display(i)

    return ce_losses.avg, dice_losses.avg

@torch.no_grad()
def validate(
    val_loader: torch.utils.data.DataLoader, model: nn.Module, select_id: List[int]
) -> Tuple[float, float]:

    di = torch.zeros(args.eval_classes).long().cuda()
    dc = torch.zeros(args.eval_classes).long().cuda()
    ii = torch.zeros(args.eval_classes).long().cuda()
    ic = torch.zeros(args.eval_classes).long().cuda()

    eval_class_idx = []
    if args.eval_classes == 3:
        eval_class_idx = [0, 1, 2]
    elif args.eval_classes == 2:
        eval_class_idx = [0, 1]
    elif args.eval_classes == 1:
        eval_class_idx = [0]
    else:
        raise RuntimeError(
            f"args.eval_classes should be 5 or 4, get args.eval_classes = {args.eval_classes}"
        )

    # switch to evaluate mode
    model.eval()
    for i, (inputs, target) in enumerate(val_loader):
        inputs = inputs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if target.shape[1] > 1:
            target = target[:, -1, :, :]

        _, dice_target = batch_process_label(target)

        if args.model == 'cnn':
            choice = select_id
            output = model(inputs, choice)
        else:
            depth_idx, attn_idx, mlp_idx = select_id['depth'], select_id['heads'], select_id['mlp_ratios']
            output = model(inputs, depth_idx, attn_idx, mlp_idx)
        
        _, dice_intersection, dice_cardinality = dice_score(
            output, dice_target, eval_class_idx
        )
        _, iou_intersection, iou_cardinality = iou_score(
            output, dice_target, eval_class_idx
        )
        di += dice_intersection.long()
        dc += dice_cardinality.long()
        ii += iou_intersection.long()
        ic += iou_cardinality.long()

    di, dc, ii, ic = torch_dist_sum(int(os.environ["LOCAL_RANK"]), di, dc, ii, ic)

    for i, class_idx in enumerate(eval_class_idx):
        print(f"dice class{class_idx}: {(di[i] / dc[i]).item()}")

    for i, class_idx in enumerate(eval_class_idx):
        print(f"iou class{class_idx}: {(ii[i] / ic[i]).item()}")

    dice = torch.sum(di / dc / args.eval_classes)
    iou = torch.sum(ii / ic / args.eval_classes)
    return dice.item(), iou.item()


def main():
    """
    Initialize dataset, model and training settings
    """
    init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    if rank != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    cudnn.benchmark = True

    world_size = dist.get_world_size()
    bs = args.bs // world_size  # 256 // 2 = 128
    args.lr *= args.bs / 256

    # create model and load supernet weights
    print("=> creating model %s" % args.arch)
    if args.model == 'cnn':
        model = archs.__dict__[args.arch](
            num_classes=args.num_classes,
            layers=args.layers,
            input_channels=3,
        )
        in_features = model.final.in_channels
        model.final = nn.Conv2d(in_features, args.num_classes, kernel_size=1)
    else:
        model = archs.__dict__[args.arch](
            img_size=args.img_size,
            num_classes=args.num_classes,
            depth_list=[12,13,14],
            num_heads_list=[3,4,6,8],
            mlp_ratio_list=[3,4,5],
        )
    
    model_without_ddp = model.cuda()
    if bs <= 8:
        model_without_ddp = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
    model = torch.nn.parallel.DistributedDataParallel(
        model_without_ddp, device_ids=[local_rank],find_unused_parameters=True
    )
    model = torch.compile(model)
    
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()}")

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        nesterov=True,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if not os.path.exists("checkpoints") and rank == 0:
        os.makedirs("checkpoints")

    if args.finetune_ckpt is not None:
        finetune_ckpt = "checkpoints/{}.pth.tar".format(args.finetune_ckpt)
        finetune_ckpt = torch.load(finetune_ckpt, map_location="cpu")
        # add new classfier for finetune classes
        current_ckpt = model.state_dict()
        loaded_ckpt = finetune_ckpt["model"]
        new_state_dict={k:v if v.size()==current_ckpt[k].size()  else  current_ckpt[k] for k,v in zip(current_ckpt.keys(), loaded_ckpt.values())}
        model.load_state_dict(new_state_dict, strict=False)
        # optimizer.load_state_dict(finetune_ckpt["optimizer"])
        # scaler.load_state_dict(finetune_ckpt["scaler"])

    checkpoint_path = "checkpoints/{}.pth.tar".format(args.checkpoint)
    print("checkpoint_path:", checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_dice = checkpoint["best_dice"]
        best_iou = checkpoint["best_iou"]
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0
        best_dice = 0
        best_iou = 0

    # load datasets
    if args.dataset == 'cancerinst':
        train_image_path = '/home/qinghua_mao/work/pathology/cancer-inst/train/train_images.npy'
        train_mask_path = '/home/qinghua_mao/work/pathology/cancer-inst/train/train_masks.npy'
        train_dataset = CancerInst(train_image_path, train_mask_path)
    else:
        if args.dataset == 'cryonuseg':
            dataset_path = "/home/qinghua_mao/work/pathology/CryoNuSeg"
        else:
            dataset_path = "/home/qinghua_mao/work/pathology/bcss/BCSS"

        train_image_dir = os.path.join(dataset_path, "train")
        train_mask_dir = os.path.join(dataset_path, "train_mask")

        if args.train_full:
            train_datafile = f"./dataset_creation/splits_{args.img_size}/full.json"
        else:
            train_datafile = (
                f"./dataset_creation/splits_{args.img_size}/train_{args.split}.json"
            )

        train_transforms = build_transform(is_train=True, args=args)
        mask_transforms = build_mask_transform(args=args)
        train_dataset = BCSSDataset(image_dir=train_image_dir,
                        mask_dir=train_mask_dir,
                        image_transforms=train_transforms,
                        mask_transforms=mask_transforms)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        persistent_workers=True,
    )

    if args.train_full:
        val_loader = None
    else:
        if args.dataset == 'cancerinst':
            val_image_path = '/home/qinghua_mao/work/pathology/cancer-inst/val/val_images.npy'
            val_mask_path = '/home/qinghua_mao/work/pathology/cancer-inst/val/val_masks.npy'
            val_dataset = CancerInst(val_image_path, val_mask_path)
        else:
            val_image_dir = os.path.join(dataset_path, "val")
            val_mask_dir = os.path.join(dataset_path, "val_mask")
            val_transforms = build_transform(is_train=False, args=args)
            val_dataset = BCSSDataset(image_dir=val_image_dir,
                            mask_dir=val_mask_dir,
                            image_transforms=val_transforms,
                            mask_transforms=mask_transforms)

        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=bs,
            shuffle=(val_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler,
            drop_last=False,
            persistent_workers=True,
        )
    print(len(train_dataset))
    print(len(val_dataset))

    # convert number into index
    kernel_map = {'3':0, '5':1, '7':2, 'id':3}
    num_heads_map = {'3':0, '4':1, '6':2, '8':3}
    mlp_ratio_map = {'3':0, '4':1, '5':2}
    depth_map = {'12':0, '13':1, '14':2}

    if args.model == 'cnn':
        select_id = [kernel_map[str(c)] for c in candidate]
    else:
        depth = depth_map[str(candidate['depth'])]
        heads = [num_heads_map[str(c)] for c in candidate['heads']]
        mlp_ratios = [mlp_ratio_map[str(c)] for c in candidate['mlp_ratios']]
        select_id = {'depth':depth, 'heads':heads, 'mlp_ratios':mlp_ratios} 

    if args.eval_only:
        val_dice, val_iou = validate(val_loader, model_without_ddp, select_id)
        print(
            "Epoch: {} val_dice {:.4f} - val_iou {:.4f}".format(
                start_epoch - 1, val_dice, val_iou
            )
        )
    else:
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            _, _ = train(train_loader, model, optimizer, scaler, epoch, select_id)
            if val_loader is not None:
                val_dice, val_iou = validate(val_loader, model_without_ddp, select_id)
                best_dice = max(best_dice, val_dice)
                best_iou = max(best_iou, val_iou)

                print(
                    "Epoch: {} val_dice {:.4f} - val_iou {:.4f} - best_dice {:.4f} - best_iou {:.4f}".format(
                        epoch, val_dice, val_iou, best_dice, best_iou
                    )
                )

            if rank == 0:
                state_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_dice": float(best_dice),
                    "best_iou": float(best_iou),
                    "epoch": epoch + 1,
                }
                torch.save(state_dict, checkpoint_path)
    destroy_process_group()

def seg_mask(args):
    import matplotlib.pyplot as plt
    import torchvision
    """
    Initialize dataset, model and training settings
    """
    set_seed(args.seed)
    init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    if rank != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    cudnn.benchmark = True

    world_size = dist.get_world_size()
    bs = args.bs // world_size  # 256 // 2 = 128
    args.lr *= args.bs / 256

    # create model and load supernet weights
    print("=> creating model %s" % args.arch)
    if args.model == 'cnn':
        model = archs.__dict__[args.arch](
            num_classes=args.num_classes,
            layers=args.layers,
            input_channels=3,
        )
        in_features = model.final.in_channels
        model.final = nn.Conv2d(in_features, args.num_classes, kernel_size=1)
    else:
        model = archs.__dict__[args.arch](
            img_size=args.img_size,
            num_classes=args.num_classes,
            depth_list=[12,13,14],
            num_heads_list=[3,4,6,8],
            mlp_ratio_list=[3,4,5],
        )

    model_without_ddp = model.cuda()
    if bs <= 8:
        model_without_ddp = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
    model = torch.nn.parallel.DistributedDataParallel(
        model_without_ddp, device_ids=[local_rank],find_unused_parameters=True
    )
    model = torch.compile(model)

    checkpoint_path = "checkpoints/{}.pth.tar".format(args.checkpoint)
    print("checkpoint_path:", checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    

    # load datasets
    if args.dataset == 'cancerinst':
        train_image_path = '/home/qinghua_mao/work/pathology/cancer-inst/train/train_images.npy'
        train_mask_path = '/home/qinghua_mao/work/pathology/cancer-inst/train/train_masks.npy'
        train_dataset = CancerInst(train_image_path, train_mask_path)
    else:
        if args.dataset == 'cryonuseg':
            dataset_path = "/home/qinghua_mao/work/pathology/CryoNuSeg"
        else:
            dataset_path = "/home/qinghua_mao/work/pathology/bcss/BCSS"

    if args.train_full:
        val_loader = None
    else:
        if args.dataset == 'cancerinst':
            val_image_path = '/home/qinghua_mao/work/pathology/cancer-inst/val/val_images.npy'
            val_mask_path = '/home/qinghua_mao/work/pathology/cancer-inst/val/val_masks.npy'
            val_dataset = CancerInst(val_image_path, val_mask_path)
        else:
            val_image_dir = os.path.join(dataset_path, "val")
            val_mask_dir = os.path.join(dataset_path, "val_mask")
            val_transforms = build_transform(is_train=False, args=args)
            mask_transforms = build_mask_transform(args=args)
            val_dataset = BCSSDataset(image_dir=val_image_dir,
                            mask_dir=val_mask_dir,
                            image_transforms=val_transforms,
                            mask_transforms=mask_transforms)

    # print(len(train_dataset))
    print(len(val_dataset))

    # convert number into index
    kernel_map = {'3':0, '5':1, '7':2, 'id':3}
    num_heads_map = {'3':0, '4':1, '6':2, '8':3}
    mlp_ratio_map = {'3':0, '4':1, '5':2}
    depth_map = {'12':0, '13':1, '14':2}

    if args.model == 'cnn':
        choice = [kernel_map[str(c)] for c in candidate]
    
    model.eval()
    h = 2
    w = 5

    random_index = [0,1,2,3,4,5,6,7,8,9]
    index = 0
    fig, ax = plt.subplots(w, h, figsize=(h*3, w*3))
    for i in range(w):
        for j in range(h):
            orig_image, true_mask = val_dataset[random_index[index]]
            orig_image = orig_image.unsqueeze(0).to(torch.float32)
            
            if true_mask.shape[1] > 1:
                true_mask = true_mask[-1, :, :]
            true_ones = torch.sum(true_mask == 1)
            true_zeros = torch.sum(true_mask == 0)
            if true_zeros == 0 or true_ones == 0:
                continue
            # output = model(orig_image, choice)
            with torch.no_grad():
                output = model(orig_image, choice)
                output = nn.functional.softmax(output, dim=1)
            image = orig_image.squeeze(0).to(torch.uint8)
            # mask = output > 0
            # 检查mask中的所有元素是否全为1
            # if torch.all(output == 1) or torch.all(output == 0):
            #     print(f"Output at index {index} is all ones or zeros.")
            # else:
            #     print(f"Output at index {index} is not all ones or zeros.")
            max_val, _ = torch.max(output, dim=1, keepdim=True)
            mask = (output == max_val)[:, [0,1]]
            print(mask.shape)

            mask_pixel = output.argmax(dim=1).squeeze().to(torch.bool)
            # mask_pixel = output.argmax(dim=1).squeeze(0,1)
            print(mask_pixel.shape)
            true_mask_pixel = true_mask.squeeze().to(torch.bool)
            # 检查mask中的所有元素是否全为1
            # if torch.all(mask_pixel == 1) or torch.all(mask_pixel == 0):
            #     print(f"Mask at index {index} is all ones or zeros.")
            # else:
            #     print(f"Mask at index {index} is not all ones or zeros.")
            # 检查mask中的所有元素是否全为1
            if torch.all(mask_pixel == 1) or torch.all(mask_pixel == 2):
                print(f"Mask at index {index} is all ones or zeros.")
            else:
                print(f"Mask at index {index} is not all ones or zeros.")
            # plt.imsave('log-res/spos-{}-2/image{}.jpg'.format(args.dataset, index), image.permute(1,2,0).cpu().numpy())
            # plt.imsave('log-res/spos-{}-1/mask{}.jpg'.format(args.dataset, index), mask.cpu().numpy(), cmap=plt.cm.gray)
            # mask_image = torchvision.utils.draw_segmentation_masks(image, mask, 1.0, colors=['','yellow','blue'])
            mask_image = torchvision.utils.draw_segmentation_masks(image, mask.squeeze(0), 0.4, ['pink','blue'])
            true_mask_image = torchvision.utils.draw_segmentation_masks(image, true_mask_pixel, 0.4, ['blue','pink'])
            ax[i, j].imshow(mask_image.permute(1, 2, 0))
            ax[i, j].axis(False)
            plt.imsave('log-res/spos-{}-2/overlap{}.jpg'.format(args.dataset, index), mask_image.permute(1, 2, 0).cpu().numpy())
            plt.imsave('log-res/spos-{}-2/true{}.jpg'.format(args.dataset, index), true_mask_image.permute(1, 2, 0).cpu().numpy())
            index+=1
    # plt.savefig('log-res/spos-{}/vis-s0-unet-{}.jpg'.format(args.dataset, args.dataset))
    destroy_process_group()


if __name__ == "__main__":
    main()
    # seg_mask(args)