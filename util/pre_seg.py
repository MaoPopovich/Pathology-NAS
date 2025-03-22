import pandas as pd
from pathlib import Path

# We define a function to create a list of the paths of the images and masks.
def image_mask_path(image_path:str, mask_path:str, train:bool):
    print('ok')
    IMAGE_PATH = Path(image_path)
    IMAGE_PATH_LIST = sorted(list(IMAGE_PATH.glob("*.png")))

    MASK_PATH = Path(mask_path)
    MASK_PATH_LIST = sorted(list(MASK_PATH.glob("*.png")))

    data = pd.DataFrame({'Image':IMAGE_PATH_LIST, 'Mask':MASK_PATH_LIST})
    if train:
        data.to_csv('/home/qinghua_mao/work/pathology/bcss/BCSS/bcss_train.csv', index=False)
    else:
        data.to_csv('/home/qinghua_mao/work/pathology/bcss/BCSS/bcss_val.csv', index=False)


def pre_bcss():
    image_train_path = "/home/qinghua_mao/work/pathology/bcss/BCSS/train"
    mask_train_path = "/home/qinghua_mao/work/pathology/bcss/BCSS/train_mask"

    image_val_path = "/home/qinghua_mao/work/pathology/bcss/BCSS/val"
    mask_val_path = "/home/qinghua_mao/work/pathology/bcss/BCSS/val_mask"

    image_mask_path(image_train_path, mask_train_path, train=True)
    image_mask_path(image_val_path, mask_val_path, train=False)


# run python mess/prepare_datasets/prepare_cryonuseg.py

import tqdm
import os
from pathlib import Path
import gdown
import kaggle

import numpy as np
from PIL import Image


def download_dataset(ds_path):
    """
    Downloads the dataset
    """
    print('Downloading dataset...')
    # Downloading kaggle
    try:
        kaggle.api.authenticate()
    except:
        raise Exception('Please install kaggle and save credentials in ~/.kaggle/kaggle.json, '
                        'see https://github.com/Kaggle/kaggle-api')
    # CLI: kaggle datasets download -d ipateam/segmentation-of-nuclei-in-cryosectioned-he-images
    kaggle.api.dataset_download_cli('ipateam/segmentation-of-nuclei-in-cryosectioned-he-images', path=ds_path, unzip=True)


def convert():
    dataset_dir = Path('/home/qinghua_mao/work/pathology')
    ds_path = dataset_dir / 'CryoNuSeg'
    if not ds_path.exists():
        download_dataset(ds_path)

    assert ds_path.exists(), f'Dataset not found in {ds_path}'

    # create directories
    img_dir = ds_path / 'images_detectron2'
    anno_dir = ds_path / 'annotations_detectron2'
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)

    for img_path in tqdm.tqdm((ds_path / 'tissue images').glob('*.tif')):
        id = img_path.stem
        # Move image
        img = Image.open(img_path)
        img = img.convert('RGB')
        img.save(img_dir / f'{id}.png', 'PNG')

        # Open mask
        mask = Image.open(ds_path / 'Annotator 1 (biologist second round of manual marks up)' / 'mask binary' / f'{id}.png')
        # Edit annotations
        # Binary encoding: (0, 255) -> (0, 1) for binary classification
        mask = np.uint8(np.array(mask) / 255)
        # Save mask
        Image.fromarray(mask).save(anno_dir / f'{id}.png')

    print(f'Saved images and masks of {ds_path.name} dataset')

def split():
    import os
    import shutil
    from random import sample

    path_src = 'CryoNuSeg/images_detectron2'
    mask_src = 'CryoNuSeg/annotations_detectron2'
    path_dst = 'CryoNuSeg'
    train_img_dir = os.path.join(path_dst, 'train')
    train_mask_dir = os.path.join(path_dst, 'train_mask')
    val_img_dir = os.path.join(path_dst, 'val')
    val_mask_dir = os.path.join(path_dst, 'val_mask')
    train_ratio = 0.8 # 训练集占比
    val_ratio = 0.2  # 验证集占比
    # test_ratio = 0.2 # 测试集占比

    # class_dir = os.path.join(path_src, class_name)
    imgs = os.listdir(path_src)
    # masks = os.listdir(mask_src)
    train_count = int(len(imgs) * train_ratio)
    val_count = int(len(imgs) * val_ratio)
    # test_count = int(len(files) * test_ratio)
    
    # 随机选择一部分文件作为训练集
    train_imgs = sample(imgs, train_count)
    # train_masks = sample(masks, train_count)
    val_imgs = list(set(imgs) - set(train_imgs))
    # val_masks = list(set(masks) - set(train_masks))
    
    # 确保训练集的目标目录存在
    # train_class_dir = os.path.join(train_dir, class_name)
    if not os.path.exists(train_img_dir):
        os.makedirs(train_img_dir)
    if not os.path.exists(train_mask_dir):
        os.makedirs(train_mask_dir)

    
    # 将选中的文件移动到训练集目录
    for img in train_imgs:
        src_img_path = os.path.join(path_src, img)
        src_mask_path = os.path.join(mask_src, img)
        dst_img_path = os.path.join(train_img_dir, img)
        dst_mask_path = os.path.join(train_mask_dir, img)
        shutil.copy(src_img_path, dst_img_path)
        shutil.copy(src_mask_path, dst_mask_path)
    
    # 随机选择一部分文件作为验证集
    # val_files = sample(files, val_count)
    
    # 确保验证集的目标目录存在
    # val_class_dir = os.path.join(val_dir, class_name)
    if not os.path.exists(val_img_dir):
        os.makedirs(val_img_dir)
    if not os.path.exists(val_mask_dir):
        os.makedirs(val_mask_dir)
    
    # 将选中的文件移动到验证集目录
    for img in val_imgs:
        src_img_path = os.path.join(path_src, img)
        src_mask_path = os.path.join(mask_src, img)
        dst_img_path = os.path.join(val_img_dir, img)
        dst_mask_path = os.path.join(val_mask_dir, img)
        shutil.copy(src_img_path, dst_img_path)
        shutil.copy(src_mask_path, dst_mask_path)


def pre_cryo():
    convert()
    split()

def pre_cancerinst():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cv2
    import torch
    import torchvision
    from torchsummary import summary
    from sklearn.model_selection import train_test_split
    import os
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from tqdm import tqdm
    types = np.load('cancer-inst/Part 1/Images/types.npy')
    print(pd.unique(types))

    images = np.load('cancer-inst/Part 1/Images/images.npy')
    images = images.astype('int32')

    masks = np.load('cancer-inst/Part 1/Masks/masks.npy')

    print(images.shape)
    print(masks.shape)
    h = 10
    w = 3

    random_index = np.random.choice(len(images), (h*w)+1, replace=False)
    index = 0
    fig, ax = plt.subplots(w, h, figsize=(h*3, w*3))
    for i in range(w):
        for j in range(h):
            image = torch.tensor(images[random_index[index]], dtype=torch.uint8).permute(2, 0, 1)
            mask = torch.tensor(masks[random_index[index]]).permute(2, 0, 1).to(torch.bool)
            mask_image = torchvision.utils.draw_segmentation_masks(image, mask, 0.2, colors='blue')
            ax[i, j].imshow(mask_image.permute(1, 2, 0))
            ax[i, j].axis(False)
            index+=1
    plt.savefig('cancer-inst/example-2.jpg')
    # train_image, val_image, train_mask, val_mask = train_test_split(images, masks, test_size=0.2, random_state=42)
    # np.save('cancer-inst/train/train_image.npy', train_image)
    # np.save('cancer-inst/train/train_mask.npy', train_mask)
    # print('train set done!')
    # np.save('cancer-inst/val/val_image.npy', val_image)
    # np.save('cancer-inst/val/val_mask.npy', val_mask)
    # print('val set done!')

    


if __name__ == '__main__':
    pre_cancerinst()
