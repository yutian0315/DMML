from models import get_model
from config import OptInit 
import torch
from datasets import get_DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
from torch import nn

def save_images(images, labels, predictions, output_dir, batch_index):
    os.makedirs(output_dir, exist_ok=True)
    
    to_pil = transforms.ToPILImage()
    
    # 还原归一化的图像
    def unnormalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor
    
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    for i in range(images.size(0)):
        # 还原图像
        image = unnormalize(images[i].cpu(), mean, std)
        image = to_pil(image)
        
        # 标签和预测不需要还原
        label = to_pil(labels[i].cpu())
        prediction = to_pil(predictions[i].cpu())
        
        # 保存图像
        image.save(os.path.join(output_dir, f"image_{batch_index}_{i}.png"))
        label.save(os.path.join(output_dir, f"label_{batch_index}_{i}.png"))
        prediction.save(os.path.join(output_dir, f"prediction_{batch_index}_{i}.png"))

if __name__ == "__main__":

    opt = OptInit()
    opt.initialize()
    model = get_model(opt.args)
    output_dir = './saved_images'
    sig = nn.Sigmoid()

    train_dataloader, val_dataloader, test_dataloader = get_DataLoader(ROOT_PATH="D:\data\ISIC2018", args=opt.args)
    best_model_path = 'best_model.pth'
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print('已加载最佳模型。')

    for batch_index, (batch_data, batch_labels) in enumerate(test_dataloader):    
        batch_data, batch_labels = batch_data.to(opt.args.device), batch_labels.to(opt.args.device) 
        # 前向传播得到预测结果
        batch_predictions = model(batch_data)

        # 应用阈值以获得二值化预测结果（假设最后一层使用了sigmoid激活）
        batch_predictions = (batch_predictions > 0.5).float()

        # 保存图像、标签和预测结果
        save_images(batch_data, batch_labels, batch_predictions, output_dir, batch_index)
        
        print(f"已保存第{batch_index}个batch的图像。")

        break  # 跳出循环，因为我们只需要第一个batch 

