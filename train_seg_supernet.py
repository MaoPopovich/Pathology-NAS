import os
import math
import time
import torch
import argparse
import builtins
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from typing import Tuple
import backbone as archs
from thop import profile
from util.cost_func import DiceLoss, CrossEntropyAvoidNaN
from util.metrics import iou_score, dice_score
from util.meter import AverageMeter, ProgressMeter, InstantMeter
from util.torch_dist_sum import torch_dist_sum
from util.dist_init import set_seed, random_choice
from util.samplers import RASampler
from data.dataset import LMDBDataset, TinyImagenet
from torch.distributed import init_process_group, destroy_process_group
from data.augmentation import TranAugment, val_aug
import torch._dynamo
torch._dynamo.config.suppress_errors = True

parser = argparse.ArgumentParser()
# Model Settings
parser.add_argument("--port", type=int, default=23457)
parser.add_argument("--arch", type=str, default="UNet")
parser.add_argument("--model", type=str, default="vit")
parser.add_argument("--num_classes", type=int, default=200)
parser.add_argument("--eval_classes", type=int, default=200)

# Training Settings
parser.add_argument("--seed", type=int, default=0, help="training seed")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--alpha", type=float, default=0.0)
# parser.add_argument("--smoothing", type=float, default=0.0)
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
# Supernet Setting
parser.add_argument("--layers", type=int, default=9, help="number of choice_block layers")
parser.add_argument("--num_choices", type=int, default=4, help="number of choices per layer")
parser.add_argument("--use_lmdb", action='store_true', help="whether or not use LMDB dataset")

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
set_seed(args.seed)
depth_list = [12,13,14]

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
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
) -> Tuple[float, float, float]:
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
    for i, (inputs, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)

        inputs_len = len(inputs)  # data augmentation to generate two views
        for img_idx in range(inputs_len):
            inputs[img_idx] = inputs[img_idx].cuda(non_blocking=True)

        bs = inputs[0].size(0)
        inputs = torch.cat(inputs)

        adjust_learning_rate(optimizer, epoch, i, iteration_per_epoch)
        ce_target, dice_target = batch_process_label(target)  # for ce_loss and dice_loss
        data_time.update(time.time() - end)

        loss = 0
        with torch.cuda.amp.autocast(enabled=True):
            if args.model == 'cnn':
                choice = random_choice(args.num_choices, args.layers)
                outputs = model(inputs, choice)
            else:
                depth_idx = np.random.randint(3)
                attn_idx = random_choice(4, depth_list[depth_idx])
                mlp_idx = random_choice(3, depth_list[depth_idx])
                outputs = model(inputs, depth_idx, attn_idx, mlp_idx)
            for output in torch.chunk(outputs, inputs_len):
                dice_loss, _, _ = DiceLoss(output, dice_target)
                loss += dice_loss
            loss /= inputs_len

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
    val_loader: torch.utils.data.DataLoader, model: nn.Module
) -> Tuple[float, float, float]:

    di = torch.zeros(args.eval_classes).long().cuda()
    dc = torch.zeros(args.eval_classes).long().cuda()
    ii = torch.zeros(args.eval_classes).long().cuda()
    ic = torch.zeros(args.eval_classes).long().cuda()

    eval_class_idx = []
    if args.eval_classes == 2:
        eval_class_idx = [0, 1]
    elif args.eval_classes == 1:
        eval_class_idx = [1]
    else:
        raise RuntimeError(
            f"args.eval_classes should be 5 or 4, get args.eval_classes = {args.eval_classes}"
        )

    # switch to evaluate mode
    model.eval()
    for i, (inputs, target) in enumerate(val_loader):
        inputs = inputs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        _, dice_target = batch_process_label(target)

        choice = random_choice(args.num_choices, args.layers)
        output = model(inputs, choice)
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
    return dice, iou


def main():
    init_process_group(backend="nccl")

    rank = dist.get_rank()
    print("rank", rank)
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    if rank != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    cudnn.benchmark = True

    world_size = dist.get_world_size()
    bs = args.bs // world_size
    args.lr *= args.bs / 256

    if args.train_full:
        train_datafile = f"./dataset_creation/splits_{args.img_size}/full.json"
    else:
        if args.use_lmdb:
            train_datafile = (
                f"./dataset_creation/splits_{args.img_size}/train_{args.split}.json"
            )

            train_dataset = LMDBDataset(
                datafile=train_datafile,
                img_size=args.img_size,
                use_paired_rrc=args.w_rcc,
                transform=TranAugment(),
                lmdb_path=f"/home/qinghua_mao/work/pathology/dataset/lmdb_imgs_{args.img_size}",
            )
        else:
            train_dataset = TinyImagenet(
                args,
                dataset_path=f"./dataset/tiny-imagenet-200/train",
                enable_train=True,
                pretrained=False
            )
    if args.repeated_aug:
        train_sampler = RASampler(train_dataset)
    else:
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
        if args.use_lmdb:
            test_datafile = (
                f"./dataset_creation/splits_{args.img_size}/test_{args.split}.json"
            )
            val_dataset = LMDBDataset(
                datafile=test_datafile,
                img_size=args.img_size,
                use_paired_rrc=False,
                transform=val_aug,
                lmdb_path=f"/home/qinghua_mao/work/pathology/dataset/lmdb_imgs_{args.img_size}",
            )
        else:
            val_dataset = TinyImagenet(
                args,
                dataset_path=f"./dataset/tiny-imagenet-200/val", 
                enable_train=False,
                pretrained=False
            )

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

    # create model
    print("=> creating model %s" % args.arch)
    if args.model == 'cnn':
        model = archs.__dict__[args.arch](
            num_classes=args.num_classes,
            layers=args.layers,
            input_channels=3,
        )
    else:
        model = archs.__dict__[args.arch](
            depth_list=[12,13,14],
            num_heads_list=[3,4,6,8],
            mlp_ratio_list=[3,4,5]
        )

    # input = torch.randn(1, 3, args.img_size, args.img_size)
    # choice = random_choice(args.num_choices, args.layers)
    # macs, params = profile(model, inputs=(input,choice))
    # print("Flops: {}, Params: {}".format(macs / 1e9, params / 1e6))

    model_without_ddp = model.cuda()
    if bs <= 8:
        model_without_ddp = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
    model = torch.nn.parallel.DistributedDataParallel(
        model_without_ddp, device_ids=[local_rank],find_unused_parameters=True
    )
    model = torch.compile(model)
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
        model.load_state_dict(finetune_ckpt["model"])
        optimizer.load_state_dict(finetune_ckpt["optimizer"])
        scaler.load_state_dict(finetune_ckpt["scaler"])

    checkpoint_path = "checkpoints/{}.pth.tar".format(args.checkpoint)
    print("checkpoint_path:", checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_iou = checkpoint["best_iou"]
        best_dice = checkpoint["best_dice"]
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0
        best_dice = 0
        best_iou = 0

    if args.eval_only:
        val_dice, val_iou = validate(val_loader, model_without_ddp)
        print(
            "Epoch: {} val_dice {:.4f} - val_iou {:.4f}".format(
                start_epoch - 1, val_dice, val_iou
            )
        )
    else:
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            _, _ = train(train_loader, model, optimizer, scaler, epoch)
            if val_loader is not None:
                val_dice, val_iou = validate(val_loader, model_without_ddp)
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


if __name__ == "__main__":
    main()
