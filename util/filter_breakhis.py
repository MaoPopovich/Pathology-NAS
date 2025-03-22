import pandas as pd
import numpy as np
import os

# def getListOfFiles(dirName):
#     listOfFile = os.listdir(dirName)
#     allFiles = list()
#     for entry in listOfFile:
#         fullPath = os.path.join(dirName, entry)
#         if os.path.isdir(fullPath):
#             allFiles = allFiles + getListOfFiles(fullPath)
#         else:
#             allFiles.append(fullPath)
                
#     return allFiles


# benign_files = getListOfFiles('BreaKHis_v1/histology_slides/breast/benign/SOB/')
# malignent_files = getListOfFiles('BreaKHis_v1/histology_slides/breast/malignant/SOB/')

# data = pd.DataFrame(index=np.arange(0, len(benign_files)+len(malignent_files)), columns=["image", "target"])
# k=0
# for c in [0,1]:
#         if c==1:
#             for m in range(len(benign_files)):
#                 data.iloc[k]["image"] = benign_files[m]
#                 data.iloc[k]["target"] = 0
#                 k += 1
#         else:
#             for m in range(len(malignent_files)):
#                 data.iloc[k]["image"] = malignent_files[m]
#                 data.iloc[k]["target"] = 1
#                 k += 1
# print(data.shape)
# print(data.head())

class_names = ['benign', 'malignant']
data = pd.read_csv('BreaKHis_v1/Folds.csv')
data = data.rename(columns={'filename':'path'})
data['label'] = data.path.apply(lambda x: x.split('/')[3])
data['label_int'] = data.label.apply(lambda x: class_names.index(x))
data['filename'] = data.path.apply(lambda x: x.split('/')[-1])

print('Count of Benign:', data[data.label == 'benign'].label.count())
print('Count of Malignant:', data[data.label == 'malignant'].label.count())


test_df = data[data.grp == 'test']
train_df = data.drop(test_df.index).reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# split training and validation set
valid_df = train_df.sample(frac=0.2)
train_df = train_df.drop(valid_df.index).reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

# test_df['set'] = 'test'
# train_df['set'] = 'train'
# valid_df['set'] = 'valid'
data_new = pd.concat([train_df,valid_df, test_df])

data_new.to_csv('BreaKHis_v1/breakhis.csv', index=False)

print('Training set')
print(train_df.label.value_counts())

print('\nValidation set')
print(valid_df.label.value_counts())

print('\nTest set')
print(test_df.label.value_counts())


import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
class Pathology(Dataset):
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

train_set = Pathology(csv_file="BreaKHis_v1/breakhis.csv",
                    root_dir="",
                    flag="train",
                    transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)

for image, label in train_loader:
    print(image.shape, label.shape)
    break

import shutil

path_dst = 'breakhis'

df_index = pd.read_csv('BreaKHis_v1/breakhis.csv')
splits = ['train', 'val', 'test']
classes = ['benign', 'malignant']

for split in splits:
    for class_name in classes:
        if not os.path.exists(os.path.join(path_dst, split, class_name)):
            os.makedirs(os.path.join(path_dst, split, class_name))

for idx, row in df_index.iterrows():
    split = row.iloc[2]
    src_file_path = row.iloc[3]
    tumor = src_file_path.split('/')[3]
    class_name = 'benign' if 'benign' in tumor.lower() else 'malignant'
    dst_file_path = os.path.join(path_dst, split, class_name, os.path.basename(src_file_path))
    
    shutil.copy(src_file_path, dst_file_path)

def split():
    import os
    import shutil
    from random import sample

    path_src = 'gaussian_filtered_images'
    path_dst = 'diabetic'
    train_dir = os.path.join(path_dst, 'train')
    val_dir = os.path.join(path_dst, 'val')
    test_dir = os.path.join(path_dst, 'test')
    classes = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
    train_ratio = 0.6 # 训练集占比
    val_ratio = 0.2  # 验证集占比
    test_ratio = 0.2 # 测试集占比

    for class_name in classes:
        class_dir = os.path.join(path_src, class_name)
        files = os.listdir(class_dir)
        train_count = int(len(files) * train_ratio)
        val_count = int(len(files) * val_ratio)
        test_count = int(len(files) * test_ratio)
        
        # 随机选择一部分文件作为训练集
        train_files = sample(files, train_count)
        remaining_files = list(set(files) - set(train_files))
        
        # 从剩余文件中随机选择一部分作为验证集
        val_files = sample(remaining_files, val_count)
        remaining_files = list(set(remaining_files) - set(val_files))
        
        # 剩余的文件作为测试集
        test_files = remaining_files
        
        # 确保验证集的目标目录存在
        train_class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        
        # 将选中的文件移动到验证集目录
        for file in train_files:
            src_file_path = os.path.join(class_dir, file)
            dst_file_path = os.path.join(train_class_dir, file)
            shutil.copy(src_file_path, dst_file_path)
        
        # # 随机选择一部分文件作为验证集
        # val_files = sample(files, val_count)
        
        # 确保验证集的目标目录存在
        val_class_dir = os.path.join(val_dir, class_name)
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)
        
        # 将选中的文件移动到验证集目录
        for file in val_files:
            src_file_path = os.path.join(class_dir, file)
            dst_file_path = os.path.join(val_class_dir, file)
            shutil.copy(src_file_path, dst_file_path)

        # # 随机选择一部分文件作为测试集
        # test_files = sample(files, test_count)
        
        # 确保测试集的目标目录存在
        test_class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)
        
        # 将选中的文件移动到验证集目录
        for file in test_files:
            src_file_path = os.path.join(class_dir, file)
            dst_file_path = os.path.join(test_class_dir, file)
            shutil.copy(src_file_path, dst_file_path)
