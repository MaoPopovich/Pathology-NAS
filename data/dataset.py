import torch.utils.data as data
from PIL import Image
import lmdb
import pickle
import io
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import sys
import os
sys.path.append(os.getcwd())
from data.augmentation import *
from typing import List,Tuple
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import pandas as pd

class LMDBDataset(data.Dataset):
    def __init__(
        self,
        img_size: int = 224,
        datafile: str = None,
        transform: transforms.Compose = None,
        use_paired_rrc: bool = False,
        lmdb_path: str = "/home/qinghua_mao/work/pathology/dataset/lmdb_imgs_224",
    ):
        super().__init__()

        self.env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        self.samples = None
        if datafile is not None:
            with open(datafile, "r") as fp:
                self.samples = json.load(fp)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

        self.use_paired_rrc = use_paired_rrc
        self.paired_rrc = PairedRandomResizedCrop(img_size)
        self.transform = transform
        self.label_transform = transforms.ToTensor()

    def __len__(self):
        if self.samples is not None:
            return self.samples.__len__()
        return self.length

    def __getitem__(self, index):
        key = self.samples[str(index)]
        key = u'{}'.format(key).encode()

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(key)

        img_bytes, label_bytes = pickle.loads(byteflow)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        label = Image.open(io.BytesIO(label_bytes))

        if self.use_paired_rrc:
            img, label = self.paired_rrc(img, label)
        img, label = random_horizontal_flip(img, label)
        img, label = random_vertical_flip(img, label)
        img, label = random_rotate(img, label)

        if not self.transform is None:
            img = self.transform(img)

        label = torch.from_numpy(np.array(label))
        return img, label.long()


class LMDBDatasetWithList(data.Dataset):
    def __init__(
        self,
        wsi_list: List[str] = None,
        transform: transforms.Compose = None,
        lmdb_path: str = "/home/qinghua_mao/work/pathology/dataset/lmdb_imgs_224",
    ):
        super().__init__()

        self.env = lmdb.open(
            lmdb_path,
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        self.samples = wsi_list
        self.transform = transform
        self.label_transform = transforms.ToTensor()
        

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        key = self.samples[index].encode()

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(key)

        img_bytes, label_bytes = pickle.loads(byteflow)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        label = Image.open(io.BytesIO(label_bytes))

        if not self.transform is None:
            img = self.transform(img)

        label = torch.from_numpy(np.array(label))
        return img, label.long()

class Pathology(data.Dataset):
    def __init__(self, csv_file, root_dir, flag="train", transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data.grp==flag]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 3])
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx, 5]

        if self.transform:
            image = self.transform(image)

        return image, label

class BCSSDataset(data.Dataset):
    def __init__(self, image_dir:str, mask_dir:str, image_transforms=None, mask_transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # image = self.image_transforms(image)
        image = self.mask_transforms(image)
        mask = self.mask_transforms(mask)

        mask = mask.long()
        
        return image, mask

class CancerInst(data.Dataset):
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path
        images = np.load(self.image_path)
        self.images = images.astype('int32')
        self.masks = np.load(self.mask_path)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)

        mask = self.masks[idx]
        mask = torch.tensor(mask, dtype=torch.float32)
        mask = mask.permute(2, 0, 1)

        return image, mask

class TinyImagenet(data.Dataset):
    def __init__(self, dataset_path='diabetic/train', transforms=None):
        # # Normalization
        # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        #     if pretrained else transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        # # Normal transformation
        # if pretrained:
        #     train_trans = [transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224), 
        #                     transforms.ToTensor()]
        #     val_trans = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm]
        # else:
        #     train_trans = [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        #     val_trans = [transforms.ToTensor(), norm]
        
        # trans = train_trans + [norm] if enable_train else val_trans

        self.dataset = datasets.ImageFolder(dataset_path, transform=transforms)

    def __len__(self):
            return self.dataset.__len__()
    def __getitem__(self, index):
            return self.dataset.__getitem__(index)

@torch.no_grad()
def batch_process_label(label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    class0 = label == 0
    class1 = label == 1
    stack_label = torch.stack([class0, class1], dim=1)   # label for ce_loss, stack_label for dice loss
    stack_label = (stack_label > 0.5).float()
    return label.long(), stack_label

# 动态应用transform
def apply_transform(dataset, transform):
    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, transform=None):
            super().__init__()
            self.base_dataset = base_dataset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.base_dataset[index]
            if self.transform is not None:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.base_dataset)
    
    return TransformedDataset(dataset, transform)

def init_imagenet(data_dir='./dataset/tiny-imagenet-1k/train', pretrained=True, args=None):
    dataset = datasets.ImageFolder(root=data_dir)
    
    if not pretrained:
        return dataset

    train_transform = build_transform(is_train=True, args=args)
    val_transform = build_transform(is_train=False, args=args)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_size = int(0.9 * len(dataset))
    train_dataset = data.Subset(dataset, indices[:train_size])
    val_dataset = data.Subset(dataset, indices[train_size:])

    # 应用transform到各自的子集
    train_dataset = apply_transform(train_dataset, train_transform)
    val_dataset = apply_transform(val_dataset, val_transform)

    return train_dataset, val_dataset

def build_transform(is_train, args):
    resize_im = args.img_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.img_size,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
        # replace RandomResizedCropAndInterpolation with
        # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.img_size, padding=4)
        return transform
    
    t = []
    if resize_im:
        size = int((224 / 224) * args.img_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.img_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def build_mask_transform(args):
    mask_transforms = transforms.Compose([transforms.Resize((args.img_size,args.img_size)), 
                                      transforms.PILToTensor()])
    return mask_transforms
    # resize_im = args.img_size > 32
    # t = []
    # if resize_im:
    #     size = int((224 / 224) * args.img_size)
    #     t.append(
    #         transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
    #     )
    #     t.append(transforms.CenterCrop(args.img_size))
    # t.append(transforms.PILToTensor())

    # return transforms.Compose(t)
    
if __name__ == "__main__":
    # train_dir = './dataset/tiny-imagenet-200/train'
    # val_dir = './dataset/tiny-imagenet-200/val'
    # train_data = TinyImagenet(dataset_path=train_dir, enable_train=True)
    # val_data = TinyImagenet(dataset_path=val_dir, enable_train=False)

    # print(len(train_data.dataset))
    # print(len(val_data.dataset))

    # kwargs = {'num_workers': 4, 'pin_memory': True}

    # train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=256, 
    #                                                 shuffle=True, **kwargs)
    
    # val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=256, 
    #                                                 shuffle=True, **kwargs)

    # for img, label in train_data_loader:
    #     print(img.shape, label.shape)
    #     break
    
    # from augmentation import TranAugment, val_aug
    # train_dataset = LMDBDataset(
    #     img_size=1280,
    #     use_paired_rrc=False,
    #     transform=TranAugment(),
    #     lmdb_path=f"dataset/lmdb_imgs_1280",
    # )

    # import random
    # import json

    # train_fullsize = len(train_dataset)
    # train_size = int(train_fullsize * 0.2)
    # all_indices = list(range(train_fullsize))
    # train_indices = random.sample(all_indices, train_size)
    # val_indices = list(set(all_indices) - set(train_indices))
    # with open('dataset_creation/splits_1280/train_0.json', 'w') as fp:
    #     json.dump({idx: key for idx, key in enumerate(train_indices)}, fp)
    # with open('dataset_creation/splits_1280/test_0.json', 'w') as fp:
    #     json.dump({idx: key for idx, key in enumerate(val_indices)}, fp)

    import argparse
    parser = argparse.ArgumentParser()
    # Augmentation parameters
    parser.add_argument("--img_size", type=int, default=224)
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

    csv_path = "/home/qinghua_mao/work/pathology/BreaKHis_v1/breakhis.csv"
    root_dir = "/home/qinghua_mao/work/pathology"
    dataset_path = "/home/qinghua_mao/work/pathology/CryoNuSeg"
    train_image_dir = os.path.join(dataset_path, "train")
    train_mask_dir = os.path.join(dataset_path, "train_mask")

    train_transforms = build_transform(is_train=True, args=args)
    # train_set = Pathology(csv_file=csv_path,
    #                 root_dir=root_dir,
    #                 flag="train",
    #                 transform=train_transforms)
    mask_transforms = build_mask_transform(args=args)
    train_dataset = BCSSDataset(image_dir=train_image_dir,
                    mask_dir=train_mask_dir,
                    image_transforms=train_transforms,
                    mask_transforms=mask_transforms)
    
    val_image_dir = os.path.join(dataset_path, "val")
    val_mask_dir = os.path.join(dataset_path, "val_mask")
    val_transforms = build_transform(is_train=False, args=args)
    # val_set = Pathology(csv_file=csv_path,
    #                 root_dir=root_dir,
    #                 flag="valid",
    #                 transform=val_transforms)
    val_dataset = BCSSDataset(image_dir=val_image_dir,
                        mask_dir=val_mask_dir,
                        image_transforms=val_transforms,
                        mask_transforms=mask_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    for image, mask in train_loader:
        # print(image.shape, mask.shape)
        unique_values = torch.unique(image)

        if len(unique_values) == 2 and (unique_values == torch.tensor([0, 1])).all():
            print("Mask contains only 0s and 1s.")
        else:
            print("Mask contains values other than 0s and 1s.")
            print(unique_values)


    
    