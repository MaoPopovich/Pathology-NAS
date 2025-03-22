import cv2
import os
import numpy as np
import pandas as pd

def view_bcss():
    mask_dir = '/home/qinghua_mao/work/pathology/bcss/BCSS/train_mask'
    for file in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        print(mask)
        print(mask.shape)
        break


def view_pannuke():
    types = np.load('/home/qinghua_mao/work/pathology/cancer-inst/Part 1/Images/types.npy')
    print(pd.unique(types))

    images = np.load('/home/qinghua_mao/work/pathology/cancer-inst/Part 1/Images/images.npy')
    images = images.astype('int32')

    masks = np.load('/home/qinghua_mao/work/pathology/cancer-inst/Part 1/Masks/masks.npy')

    print(images.shape)
    print(masks.shape)


if __name__ == '__main__':
    view_pannuke()