import torch.nn as nn
from config import OptInit
from tqdm import tqdm
import torch
from models import get_model
import matplotlib.pyplot as plt

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
    predictions = []  # 用于存储整个epoch的预测  
    labels = []       # 用于存储整个epoch的标签  
    progress_bar = tqdm(train_dataloader, desc="Training", leave=False)
    for batch_data, batch_labels in progress_bar:  
        batch_data, batch_labels = batch_data.to(args.device), batch_labels.to(args.device)  
          
        optimizer.zero_grad()  
        outputs = model(batch_data)  # 获取原始输出（logits或未归一化的概率）  
        loss = criterion(outputs, batch_labels)  
        loss.backward()  
        optimizer.step()  
          
        # 存储预测概率（或类别索引）和标签  
        predictions.append(outputs.cpu())  # 存储预测类别索引  
        labels.append(batch_labels.cpu())  
        total_loss += loss.item()  
  
    # 将整个epoch的预测和标签转换为单个tensor  
    predictions = torch.cat(predictions, dim=0)  
    labels = torch.cat(labels, dim=0)  
      
    # 计算整个epoch的准确率  
    accuracy =  calculate_accuracy(predictions, labels) 
    # 计算平均损失  
    avg_loss = total_loss / len(train_dataloder)  
    return accuracy, avg_loss  


def validate_one_epoch(val_dataloader, model, criterion, args):  
    ### Validation ###  
    model.eval()  # 设置模型为评估模式  
    total_loss = 0.0  
    predictions = []  
    labels = []  
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        progress_bar = tqdm(val_dataloader, desc="Validation", leave=False)  
        for batch_data, batch_labels in progress_bar:  
            batch_data, batch_labels = batch_data.to(args.device), batch_labels.to(args.device)  
            outputs = model(batch_data)  # 获取预测输出  
            loss = criterion(outputs, batch_labels)  # 计算损失  
            predictions.append(outputs.cpu())  
            labels.append(batch_labels.cpu())  
            total_loss += loss.item()  
          
    # 将整个epoch的预测和标签转换为单个tensor  
    predictions = torch.cat(predictions, dim=0)  
    labels = torch.cat(labels, dim=0)  
        
    # 计算准确率  
    accuracy = calculate_accuracy(predictions, labels)  
    # 计算平均损失  
    avg_loss = total_loss / len(val_dataloader)  
    return accuracy, avg_loss  

def main(train_dataloader, val_dataloader, test_dataloader, args):

    model = get_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    best_model_path = 'best_model.pth'

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(args.epoch):
        # 训练阶段
        train_accuracy, train_loss = train_one_epoch(train_dataloader, model, optimizer, criterion, args)
        # 验证阶段
        val_accuracy, val_loss = validate_one_epoch(val_dataloader, model, criterion, args)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # 打印统计信息
        print(f'第 {epoch+1} / {args.epoch} 轮 - train_loss: {train_loss:.4f}, train_acc: {train_accuracy:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_accuracy:.4f}')
        # 保存验证集上表现最好的模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print('保存最佳模型。')

    # 加载验证集上表现最好的模型
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print('已加载最佳模型。')

    # 测试阶段
    test_accuracy, test_loss = validate_one_epoch(test_dataloader, model, criterion, args)
    print(f'测试集损失: {test_loss:.4f}, 测试集准确率: {test_accuracy:.4f}')



    # 绘制损失和准确率的折线图
    epochs = range(1, args.epoch + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.show()
    plt.savefig("result.png")

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