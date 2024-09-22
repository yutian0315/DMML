import torch.nn as nn
from config import OptInit
from tqdm import tqdm
import torch
from models import get_model
import matplotlib.pyplot as plt
from utils import Metrics

def train_one_epoch(train_dataloder, model, optimizer, criterion, args):
    ### Training ###  
    model.train()  
    total_loss = 0.0  
    total_dsc = 0.0
    progress_bar = tqdm(train_dataloader, desc="Training", leave=False)
    for batch_data, batch_labels in progress_bar:  
        batch_data, batch_labels = batch_data.to(args.device), batch_labels.to(args.device)  
          
        optimizer.zero_grad()  
        outputs = model(batch_data)  # 获取原始输出（logits或未归一化的概率）  
        loss = criterion.loss(outputs, batch_labels)  
        loss.backward()  
        optimizer.step()  
        dsc = criterion.dsc(outputs, batch_labels) 
        total_loss += loss.item()  
        total_dsc  += dsc.item() 
    # 计算平均损失  
    avg_loss = total_loss / len(train_dataloder)
    avg_dsc = total_dsc / len(train_dataloder)  
    return avg_dsc, avg_loss  


def validate_one_epoch(val_dataloader, model, criterion, args):  
    ### Validation ###  
    model.eval()  # 设置模型为评估模式  
    total_loss = 0.0  
    total_dsc = 0.0
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源  
        for batch_data, batch_labels in val_dataloader:  
            batch_data, batch_labels = batch_data.to(args.device), batch_labels.to(args.device)  
            outputs = model(batch_data)  # 获取预测输出  
            loss = criterion.loss(outputs, batch_labels)  # 计算损失  
            dsc = criterion.dsc(outputs, batch_labels) 
            total_loss += loss.item()  
            total_dsc  += dsc.item()

        
        # 计算平均损失  
        avg_loss = total_loss / len(val_dataloader) 
        avg_dsc = total_dsc / len(val_dataloader)   
        return avg_dsc, avg_loss  

def main(train_dataloader, val_dataloader, test_dataloader, args):

    model = get_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = Metrics()

    best_val_dsc = 0.0
    best_model_path = 'best_model.pth'

    train_losses, val_losses = [], []
    train_dscs, val_dscs = [], []

    for epoch in range(args.epoch):
        # 训练阶段
        train_dsc, train_loss = train_one_epoch(train_dataloader, model, optimizer, criterion, args)
        # 验证阶段
        val_dsc, val_loss = validate_one_epoch(val_dataloader, model, criterion, args)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dscs.append(train_dsc)
        val_dscs.append(val_dsc)

        # 打印统计信息
        print(f'第 {epoch+1} / {args.epoch} 轮 - train_loss: {train_loss:.4f}, train_acc: {train_dsc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_dsc:.4f}')
        # 保存验证集上表现最好的模型
        if val_dsc > best_val_dsc:
            best_val_dsc = val_dsc
            torch.save(model.state_dict(), best_model_path)
            print('保存最佳模型。')

    # 加载验证集上表现最好的模型
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print('已加载最佳模型。')

    # 测试阶段
    test_dsc, test_loss = validate_one_epoch(test_dataloader, model, criterion, args)
    print(f'测试集损失: {test_loss:.4f}, 测试集dice相似度: {test_dsc:.4f}')



    # 绘制损失和准确率的折线图
    # epochs = range(1, args.epoch + 1)

    # plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    # plt.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    # plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Training and Validation Loss')

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, train_accuracies, label='Training Accuracy', linewidth=2)
    # plt.plot(epochs, val_accuracies, label='Validation Accuracy', linewidth=2)
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.title('Training and Validation Accuracy')

    # plt.tight_layout()
    # plt.show()
    # plt.savefig("result.png")

if __name__ == "__main__":
    from datasets import  get_DataLoader
    import platform
    #  参数初始化
    opt = OptInit()
    opt.initialize()
    system_name = platform.system()

    if system_name == "Windows":
        ROOT = "D:\data\ISIC2018"
    else:
        ROOT = "/home/maojunbin/data/ISIC2018"
    train_dataloader, val_dataloader, test_dataloader = get_DataLoader(ROOT, opt.args)
    main(train_dataloader, val_dataloader, test_dataloader, opt.args)
    print("finish")