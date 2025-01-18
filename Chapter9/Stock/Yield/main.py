import torch.nn as nn
from config import OptInit
from tqdm import tqdm
import torch
from models import get_model
import matplotlib.pyplot as plt
import torch.nn.functional as F  
import numpy as np

plt.rcParams['font.family'] = 'Microsoft YaHei'  
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12

def calculate_accuracy(predictions, labels):

    # 获取预测的类别索引
    _, predicted_labels = torch.max(predictions, 1)
    # 将one-hot编码的标签转换为类索引
    _, true_labels = torch.max(labels, 1)
    # 计算正确预测的数量
    correct_predictions = (predicted_labels == true_labels).sum().item()
    # 计算总样本数
    total_samples = labels.size(0)
    # 计算准确率
    accuracy = correct_predictions / total_samples
    return accuracy

def train_one_epoch(train_dataloder, model, optimizer, criterion, args):
    ### Training ###  
    model.train()  
    total_loss = 0.0   
    progress_bar = tqdm(train_dataloader, desc="Training", leave=False)
    for batch_data, batch_labels in progress_bar:  
        batch_data, batch_labels = batch_data.squeeze(1).to(args.device), batch_labels.unsqueeze(1).to(args.device)  
        optimizer.zero_grad()  
        outputs = model(batch_data)[1,:,:] 
        loss = criterion(outputs, batch_labels)  
        loss.backward()  
        optimizer.step()  
        total_loss += loss.item()   
    avg_loss = total_loss / len(train_dataloder)  
    return avg_loss  

def test_one_epoch(val_dataloader, model, criterion, args):  
    ### Test ###  
    model.eval()  # 设置模型为评估模式  
    total_loss = 0.0  
    predictions = []  
    labels = []  
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        progress_bar = tqdm(val_dataloader, desc="Validation", leave=False)  
        for batch_data, batch_labels in progress_bar:  
            batch_data, batch_labels = batch_data.squeeze(1).to(args.device), batch_labels.unsqueeze(1).to(args.device) 
            outputs = model(batch_data)[1,:,:]   # 获取预测输出  
            loss = criterion(outputs, batch_labels)  # 计算损失  
            total_loss += loss.item()  
            predictions.append(outputs.squeeze(1))
            labels.append(batch_labels.squeeze(1))
    avg_loss= total_loss / len(val_dataloader)
    return avg_loss, predictions, labels

def plot_loss(train_losses):
    plt.figure(figsize=(10, 6))  # 设置图形的大小
    plt.plot(train_losses, color='lightgreen', linewidth=2.5, marker='o', markersize=5, markerfacecolor='pink', label='训练集')  # 使用深绿色线和粉色标记
    plt.title('训练集损失', fontsize=16, fontweight='bold')  # 设置标题字体大小和加粗
    plt.xlabel('训练轮次', fontsize=14, fontweight='bold')  # 设置x轴标签字体大小和加粗
    plt.ylabel('损失', fontsize=14, fontweight='bold')  # 设置y轴标签字体大小和加粗
    plt.xticks(range(len(train_losses)), fontsize=12)  # 设置x轴刻度为整数和字体大小
    plt.yticks(fontsize=12)  # 设置y轴刻度字体大小
    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线，设置样式和透明度
    plt.legend(fontsize=12)  # 显示图例并设置字体大小
    plt.tight_layout()  # 自适应布局
    plt.show()  # 显示图形


def main(train_dataloader, test_dataloader, args, close_max, close_min):
    model = get_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()
    best_model_path = 'best_model.pth'
    train_losses = []
    Min_loss = 999
    count = 0
    for epoch in range(args.epoch):
        # 训练阶段
        train_loss = train_one_epoch(train_dataloader, model, optimizer, criterion, args)
        train_losses.append(train_loss)
        # 打印统计信息
        print(f'第 {epoch+1} / {args.epoch} 轮 - train_loss: {train_loss:.6f}, earlystop: {count}')
        # 保存验证集上表现最好的模型
        if train_loss < Min_loss:
            Min_loss = train_loss
            count = 0
            torch.save(model.state_dict(), best_model_path)
            print('保存最佳模型。')
        else: 
            count += 1
            if count == 5:
                break
    # 加载验证集上表现最好的模型
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print('已加载最佳模型。')

    # 测试阶段
    test_loss, predictions, labels = test_one_epoch(test_dataloader, model, criterion, args)
    print(f'测试集损失: {test_loss:.5f}')

    predictions = torch.cat(predictions, dim=0) 
    labels = torch.cat(labels, dim=0) 
    predictions = predictions.cpu().numpy() * (close_max - close_min) + close_min
    labels = labels.cpu().numpy() * (close_max - close_min) + close_min

    plot_loss(train_losses)

    x = range(len(predictions))
    # 创建一个图形和子图  
    plt.figure()  
    # 绘制第一条曲线  
    plt.plot(x, predictions, label='预测值', marker='o')  # marker='o'表示用圆圈标记数据点  
    
    # 绘制第二条曲线  
    plt.plot(x, labels, label='真实值', marker='^')  # marker='x'表示用叉号标记数据点  
    
    # 添加图例  
    plt.legend()  
    # 添加标题和轴标签  
    plt.title('股票价格预测值和真实值的对比')  
    plt.xlabel('x')  
    plt.ylabel('y')  
    
    # 显示网格线（可选）  
    plt.grid(True)  
    # 显示图形  
    plt.show()

if __name__ == "__main__":

    from datasets import  get_DataLoader
    import platform
    #  参数初始化
    opt = OptInit()
    opt.initialize()

    # USER_TOKEN = 'ad470017ef0a1f4d9613fb95712158e61352b39a258ec63cdf9a17cd'
    DATA_PATH = "data\\shanghai_index_data.csv"
    close_max, close_min, train_dataloader, test_dataloader = get_DataLoader(DATA_PATH, opt.args)
    main(train_dataloader, test_dataloader, opt.args, close_max, close_min)
    print("finish")