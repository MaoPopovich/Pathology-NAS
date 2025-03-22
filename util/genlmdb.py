import os
import sys
import six
import string
import argparse

import lmdb
import pickle
import msgpack
import tqdm
from PIL import Image

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
# This segfaults when imported before torch: https://github.com/apache/arrow/issues/2637
import pyarrow as pa


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        targetbuf = unpacked[1]
        buf = six.BytesIO()
        buf.write(targetbuf)
        buf.seek(0)
        target = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)


def folder2lmdb(img_dir, outpath, write_frequency=5000):
    directory = os.path.expanduser(img_dir)
    print("Loading dataset from %s" % directory)

    lmdb_path = os.path.expanduser(outpath)
    isdir = os.path.isdir(lmdb_path)

    # get image paths and label mask paths
    label_dir = img_dir.replace('imgs','labels')
    img_paths = sorted([os.path.join(img_dir, filename) for filename in os.listdir(img_dir)])
    label_paths = sorted([os.path.join(label_dir, filename) for filename in os.listdir(label_dir)])

    # check if the number of images and label masks are the same
    assert len(img_paths) == len(label_paths)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx in range(len(img_paths)):
        with open(img_paths[idx], 'rb') as f:
            img_bytes = f.read()
        with open(label_paths[idx], 'rb') as f:
            label_bytes = f.read()
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((img_bytes, label_bytes)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(img_paths)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Path to original image dataset folder")
    parser.add_argument("-o", "--outpath", help="Path to output LMDB file")
    args = parser.parse_args()
    folder2lmdb(args.dataset, args.outpath)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
	# 			 std=[0.229, 0.224, 0.225])

    # train_dataset = ImageFolderLMDB(
    #     './dataset/lmdb_imgs_224',
    #     transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    #     ]))
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # for i, (input, target) in enumerate(train_loader):
    #     print(input.size())
    #     print(target.size())
    #     break