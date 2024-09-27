import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import torch

class SkinClassDataset(Dataset):
    def __init__(self, GroundTruth_PATH, Input_PATH, transform=None):
        self.groundtruth = pd.read_csv(GroundTruth_PATH) # 读取CSV文件，存储图像名称和对应的标签
        self.input_path = Input_PATH # 输入图像文件夹的路径
        self.transform = transform # 图像变换操作
        self.label = self.groundtruth.iloc[:, 1:].values.tolist() # 获取标签列表，假设标签在CSV文件的第2列及之后的列中

    def __len__(self):
        return len(self.groundtruth)

    def __getitem__(self, index):
        image_name = self.groundtruth['image'].iloc[index] # 获取图像的文件名
        image_path = os.path.join(self.input_path, image_name+'.jpg') # 构建图像的完整路径
        image = Image.open(image_path) # 打开图像
        label = self.label[index]  # 获取对应的标签
        if self.transform:
            image = self.transform(image)  # 对图像进行变换
        label = torch.tensor(label) # 将标签转换为Tensor
        return image, label

def get_DataLoader(ROOT_PATH, args): # 创建数据读取器 DataLoader

    # 数据路径拼接
    train_groundtruth_path = os.path.join(ROOT_PATH, 'ISIC2018_Task3_Training_GroundTruth', 'ISIC2018_Task3_Training_GroundTruth.csv')
    train_data_path = os.path.join(ROOT_PATH, 'ISIC2018_Task3_Training_Input')

    val_groundtruth_path = os.path.join(ROOT_PATH, 'ISIC2018_Task3_Validation_GroundTruth', 'ISIC2018_Task3_Validation_GroundTruth.csv')
    val_data_path = os.path.join(ROOT_PATH, 'ISIC2018_Task3_Validation_Input')

    test_groundtruth_path = os.path.join(ROOT_PATH, 'ISIC2018_Task3_Test_GroundTruth', 'ISIC2018_Task3_Test_GroundTruth.csv')
    test_data_path = os.path.join(ROOT_PATH, 'ISIC2018_Task3_Test_Input')

    # 创建数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 自定义数据集创建
    train_dataset = SkinClassDataset(GroundTruth_PATH=train_groundtruth_path, Input_PATH=train_data_path, transform=train_transform)
    val_dataset = SkinClassDataset(GroundTruth_PATH=val_groundtruth_path, Input_PATH=val_data_path, transform=test_transform)
    test_dataset = SkinClassDataset(GroundTruth_PATH=test_groundtruth_path, Input_PATH=test_data_path, transform=test_transform)
    
    # 创建数据读取器
    train_dataloder = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloder = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_dataloder = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

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

