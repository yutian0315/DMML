import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

class myToTensor:
    def __init__(self, dtype = torch.float32):
        self.dtype = dtype
        self.totensor = transforms.ToTensor()
    def __call__(self, data):
        image, mask = data
        return self.totensor(image), self.totensor(mask)

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        return (TF.resize(image, [self.size_h, self.size_w], antialias=True),
                TF.resize(mask, [self.size_h, self.size_w], antialias=True, interpolation=InterpolationMode.NEAREST))

class myNormalize:
    def __init__(self, data_name='isic18', train=True):
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    def __call__(self, data):
        img, msk = data
        img = self.normalize(img)
        return img, msk

class SkinSegDataset(Dataset):
    def __init__(self, images_dir, labels_dir, data_transform=None, label_transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.image_files = sorted(os.listdir(images_dir))[1:-1]  
        self.label_files = sorted(os.listdir(labels_dir))[1:-1]  

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_name = self.label_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, label_name)

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")  

        if self.data_transform:
            image, label = self.data_transform((image, label))
        return image, label

def get_DataLoader(ROOT_PATH, args): # 创建数据读取器 DataLoader

    # 数据路径拼接
    train_input = os.path.join(ROOT_PATH, 'ISIC2018_Task1-2_Training_Input')
    val_input = os.path.join(ROOT_PATH, 'ISIC2018_Task1-2_Validation_Input')
    test_input = os.path.join(ROOT_PATH, 'ISIC2018_Task1-2_Test_Input')

    train_label = os.path.join(ROOT_PATH, 'ISIC2018_Task1_Training_GroundTruth')
    val_label = os.path.join(ROOT_PATH, 'ISIC2018_Task1_Validation_GroundTruth')
    test_label = os.path.join(ROOT_PATH, 'ISIC2018_Task1_Test_GroundTruth')

    # 创建数据预处理
    train_data_transform = transforms.Compose([
        myResize(args.resize, args.resize),
        myToTensor(),
        myNormalize(),
    ])

    test_data_transform = transforms.Compose([
        myResize(args.resize, args.resize),
        myToTensor(),
        myNormalize(),
    ])

    # 自定义数据集创建
    train_dataset = SkinSegDataset(images_dir=train_input, labels_dir=train_label, data_transform=train_data_transform)
    val_dataset = SkinSegDataset(images_dir=val_input, labels_dir=val_label, data_transform=test_data_transform)
    test_dataset = SkinSegDataset(images_dir=test_input, labels_dir=test_label, data_transform=test_data_transform)
    
    # 创建数据读取器
    train_dataloder = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloder = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloder = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_dataloder, val_dataloder, test_dataloder

if __name__ == "__main__":
    from config import OptInit
    from tqdm import tqdm

    ROOT_PATH = 'D:\data\ISIC2018'
    opt = OptInit()
    opt.initialize()
    train_dataloder, val_dataloder, test_dataloder = get_DataLoader(ROOT_PATH, opt.args)
    for image, label in tqdm(train_dataloder):
        pass

