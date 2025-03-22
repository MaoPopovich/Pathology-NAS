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
from data.dataset import LMDBDataset, init_imagenet, Pathology, build_transform, TinyImagenet
from torch.distributed import init_process_group, destroy_process_group
from data.augmentation import TranAugment, val_aug
import torch._dynamo
torch._dynamo.config.suppress_errors = True

parser = argparse.ArgumentParser()

# Supernet Setting
parser.add_argument("--layers", type=int, default=20, help="number of choice_block layers in CNNs or operations in ViT")
parser.add_argument("--num_choices", type=int, default=4, help="number of choices per layer in CNNs or per op in ViT")
parser.add_argument("--use_lmdb", action='store_true', help="whether or not use LMDB dataset")

# model settings
parser.add_argument('--model4task', type=str, default='unet4cls', required=True)
parser.add_argument('--arch', type=str, default="ResNet")
parser.add_argument("--model", type=str, default="vit")
parser.add_argument("--num_classes", type=int, default=200)
parser.add_argument("--dataset", type=str, default='breakhis')
parser.add_argument("--ft_epochs", type=int, default=10)

# training settings
parser.add_argument("--seed", type=int, default=0, help="training seed")
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

# * Mixup params
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

args = parser.parse_args()

shuffle_map = {
    'breakhis':{10: [7, 5, 7, 5, 7, 7, 5, 7, 7, 7, 5, 7, 7, 5, 7, 5, 7, 7, 7, 7], 30: [7, 7, 7, 5, 7, 5, 7, 7, 3, 7, 7, 5, 7, 7, 7, 5, 7, 5, 7, 7], 40: [7, 7, 7, 5, 7, 7, 5, 7, 7, 5, 7, 7, 5, 7, 7, 5, 7, 7, 7, 5]},
    'diabetic':{10: [7, 7, 7, 7, 5, 7, 7, 7, 5, 7, 7, 7, 5, 7, 7, 7, 5, 7, 7, 7], 30: [7, 3, 7, 3, 7, 5, 3, 7, 5, 3, 7, 3, 5, 7, 3, 5, 7, 3, 5, 7], 40: [7, 5, 5, 3, 5, 7, 7, 3, 5, 7, 7, 3, 3, 7, 7, 5, 3, 7, 5, 5]}
}
vit_map = {
    'breakhis':{10: {'depth': 14, 'heads': [8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6], 'mlp_ratios': [4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5]},
                30: {'depth': 14, 'heads': [3, 4, 6, 8, 8, 6, 4, 3, 6, 8, 4, 6, 3, 8], 'mlp_ratios': [5, 5, 5, 5, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3]},
                40: {'depth': 14, 'heads': [4, 6, 4, 6, 8, 8, 4, 6, 4, 6, 8, 8, 4, 6], 'mlp_ratios': [4, 5, 4, 5, 3, 3, 4, 5, 4, 5, 3, 3, 4, 5]}},
    'diabetic':{10: {'depth': 14, 'heads': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], 'mlp_ratios': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]},
                20: {'depth': 14, 'heads': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 'mlp_ratios': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]},
                30: {'depth': 14, 'heads': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 'mlp_ratios': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]},
                40: {'depth': 14, 'heads': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], 'mlp_ratios': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}}
}

if args.model == 'cnn':
    candidate = shuffle_map[args.dataset][args.ft_epochs]
else:
    candidate = vit_map[args.dataset][args.ft_epochs] # temp1


@torch.no_grad()
def batch_process_label(label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    class0 = label == 0
    class1 = label == 1
    stack_label = torch.stack([class0, class1], dim=1)   # label for ce_loss, stack_label for dice loss
    stack_label = (stack_label > 0.5).float()
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
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    mixup_fn: Mixup,
    select_id: List[int]
) -> Tuple[float, float]:
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    curr_lr = InstantMeter("LR", ":6.5f")
    ce_losses = AverageMeter("CE", ":.4e")
    train_acc = AverageMeter("ACC", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, curr_lr, ce_losses, train_acc],
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
        target = target.cuda(non_blocking=True)
        if input.size(0) == 2: # For classification task, data augmentation is dismissed. we only use first image of pathology datasets
            input = input[0]

        input= input.cuda(non_blocking=True)

        bs = input.size(0)
        input = input.float() # convert ByteTensor to FloatTensor

        adjust_learning_rate(optimizer, epoch, i, iteration_per_epoch)
        ce_target, dice_target = batch_process_label(target)  # for ce_loss and dice_loss
        data_time.update(time.time() - end)

        if mixup_fn is not None:
            input, ce_target = mixup_fn(input, ce_target)
        # print(ce_target.shape)
        loss = 0
        with torch.cuda.amp.autocast(enabled=True):
            if args.model == 'cnn':
                # choice = random_choice(args.num_choices, args.layers)
                choice = select_id
                output = model(input, choice)
                # inputs.shape [256,3,224,224] dice_target.shape [128,2,224,224]  ce_target.shape [128,224,224] outputs.shape # [256,2]
            else:
                # depth_idx = np.random.randint(3)
                # attn_idx = random_choice(4, depth_list[depth_idx])
                # mlp_idx = random_choice(3, depth_list[depth_idx])
                depth_idx, attn_idx, mlp_idx = select_id['depth'], select_id['heads'], select_id['mlp_ratios']
                output = model(input, depth_idx, attn_idx, mlp_idx)
            ce_loss = criterion(output, ce_target)
            loss += ce_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        curr_lr.update(optimizer.param_groups[0]["lr"])
        ce_losses.update(loss.item(), bs)

        acc = 0
        if args.num_classes > 2:
            prec1, _ = accuracy(output, ce_target, topk=(1,5))
        else:
            prec1, = accuracy(output, ce_target, topk=(1,))
        acc += prec1

        train_acc.update(acc.item(), bs)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            progress.display(i)

    return ce_losses.avg, train_acc.avg

@torch.no_grad()
def validate(
    val_loader: torch.utils.data.DataLoader, model: nn.Module, select_id
) -> Tuple[float, float]:

    val_losses = AverageMeter("CE", ":.4e")
    val_acc = AverageMeter("Acc",":.4e")
    criterion = nn.CrossEntropyLoss()

    # switch to evaluate mode
    model.eval()
    for i, (inputs, target) in enumerate(val_loader):
        inputs = inputs.cuda(non_blocking=True)
        inputs = inputs.float() # convert ByteTensor to FloatTensor
        target = target.cuda(non_blocking=True)
        ce_target, dice_target = batch_process_label(target)  # for ce_loss and dice_loss
        if args.model == 'cnn':
            # choice = random_choice(args.num_choices, args.layers)
            choice = select_id
            outputs = model(inputs, choice)
            # inputs.shape [256,3,224,224] dice_target.shape [128,2,224,224]  ce_target.shape [128,224,224] outputs.shape # [256,2]
        else:
            # depth_idx = np.random.randint(3)
            # attn_idx = random_choice(4, depth_list[depth_idx])
            # mlp_idx = random_choice(3, depth_list[depth_idx])
            depth_idx, attn_idx, mlp_idx = select_id['depth'], select_id['heads'], select_id['mlp_ratios']
            outputs = model(inputs, depth_idx, attn_idx, mlp_idx)

        loss = criterion(outputs, ce_target)

        if args.num_classes > 2:
            prec1, _ = accuracy(outputs, ce_target, topk=(1,5))
        else:
            prec1, = accuracy(outputs, ce_target, topk=(1,))

        bs = inputs.size(0)
        val_losses.update(loss.item(), bs)
        val_acc.update(prec1.item(), bs)

    return val_losses.avg, val_acc.avg

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
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, args.num_classes, bias=False)
    else:
        model = archs.__dict__[args.arch](
            img_size=args.img_size,
            num_classes=args.num_classes,
            depth_list=[12,13,14],
            num_heads_list=[3,4,6,8],
            mlp_ratio_list=[3,4,5],
        )
        embed_dim = model.head.in_features
        model.head = nn.Linear(embed_dim, args.num_classes)

    model_without_ddp = model.cuda()
    if bs <= 8:
        model_without_ddp = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
    model = torch.nn.parallel.DistributedDataParallel(
        model_without_ddp, device_ids=[local_rank],find_unused_parameters=True
    )
    model = torch.compile(model)
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()}")
    if args.mixup > 0:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        nesterov=True,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # mixup 
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

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
        best_loss = checkpoint["best_loss"]
        best_acc = checkpoint["best_acc"]
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0
        best_loss = 1e4
        best_acc = 0

    # load datasets
    csv_path = "/home/qinghua_mao/work/pathology/BreaKHis_v1/breakhis.csv"
    root_dir = "/home/qinghua_mao/work/pathology"
    diabetic_path = '/home/qinghua_mao/work/pathology/diabetic'
    if args.train_full:
        train_datafile = f"./dataset_creation/splits_{args.img_size}/full.json"
    else:
        train_datafile = (
            f"./dataset_creation/splits_{args.img_size}/train_{args.split}.json"
        )

    train_transforms = build_transform(is_train=True, args=args)

    if args.dataset == 'breakhis':
        train_dataset = Pathology(csv_file=csv_path,
                        root_dir=root_dir,
                        flag="train",
                        transform=train_transforms)
    else:
        train_dataset = TinyImagenet(dataset_path=os.path.join(diabetic_path, 'train'), transforms=train_transforms)

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
        val_transforms = build_transform(is_train=False, args=args)
        if args.dataset == 'breakhis':
            val_dataset = Pathology(csv_file=csv_path,
                            root_dir=root_dir,
                            flag="test",
                            transform=val_transforms)
        else:
            val_dataset = TinyImagenet(dataset_path=os.path.join(diabetic_path, 'test'), transforms=val_transforms)

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
        val_loss, val_acc = validate(val_loader, model_without_ddp, select_id)
        print(
            "Epoch: {} val_loss {:.4f} - val_acc {:.4f}".format(
                start_epoch - 1, val_loss, val_acc
            )
        )
    else:
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            _, _ = train(train_loader, model, criterion, optimizer, scaler, epoch, mixup_fn, select_id)
            if val_loader is not None:
                val_loss, val_acc = validate(val_loader, model_without_ddp, select_id)
                best_loss = min(best_loss, val_loss)
                best_acc = max(best_acc, val_acc)

                print(
                    "Epoch: {} val_loss {:.4f} - val_acc {:.4f} - best_loss {:.4f} - best_acc {:.4f}".format(
                        epoch, val_loss, val_acc, best_loss, best_acc
                    )
                )

            if rank == 0:
                state_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_loss": float(best_loss),
                    "best_acc": float(best_acc),
                    "epoch": epoch + 1,
                }
                torch.save(state_dict, checkpoint_path)
    destroy_process_group()



if __name__ == "__main__":
    main()