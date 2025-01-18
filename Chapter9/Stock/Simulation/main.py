import torch.nn as nn
from config import OptInit
from tqdm import tqdm
import torch
from models import get_model
import matplotlib.pyplot as plt
import torch.nn.functional as F  
import numpy as np
# import seaborn as sns

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
    plt.plot(train_losses, color='lightgreen', linewidth=2.5, marker='o', markersize=5, markerfacecolor='pink', label='Training Loss')  # 使用深绿色线和粉色标记
    plt.title('Training Loss Curve', fontsize=16, fontweight='bold')  # 设置标题字体大小和加粗
    plt.xlabel('Epochs', fontsize=14, fontweight='bold')  # 设置x轴标签字体大小和加粗
    plt.ylabel('Loss', fontsize=14, fontweight='bold')  # 设置y轴标签字体大小和加粗
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

    return train_losses, predictions, labels

    # plot_loss(train_losses)
    # x = range(len(predictions))
    # # 创建一个图形和子图  
    # plt.figure()  
    # # 绘制第一条曲线  
    # plt.plot(x, predictions, label='prediction', marker='o')  # marker='o'表示用圆圈标记数据点  
    # # 绘制第二条曲线  
    # plt.plot(x, labels, label='true value', marker='^')  # marker='x'表示用叉号标记数据点  
    # # 添加图例  
    # plt.legend()  
    # # 添加标题和轴标签  
    # plt.title('Comparison between predicted values and actual values of stock prices')  
    # plt.xlabel('x')  
    # plt.ylabel('y')  
    
    # # 显示网格线（可选）  
    # plt.grid(True)  
    # # 显示图形  
    # plt.show()

def plot_loss_curves(data, labels=None, title='训练集损失', xlabel='训练轮次', ylabel='损失'):  
    """  
    绘制训练损失曲线。  
  
    参数:  
    data (list of lists): 每个子列表表示一个模型的loss值。  
    labels (list of str, optional): 每个模型的标签，用于图例。如果未提供，则使用默认标签'Model 1', 'Model 2', ...。  
    title (str): 图形的标题。  
    xlabel (str): x轴的标签。  
    ylabel (str): y轴的标签。  
    """  
    # 如果没有提供标签，则生成默认标签  
    if labels is None:  
        labels = [f'Model {i+1}' for i in range(len(data))]  
      
    # 设置x轴（epochs），假设所有子列表长度相同  
    epochs = np.arange(1, len(data[0]) + 1)  
      
    # 设置seaborn样式为whitegrid，它类似于ggplot  
    # sns.set(style="whitegrid")  
      
    # 创建一个图形和一个轴  
    fig, ax = plt.subplots()  
      
    # 绘制每条曲线，颜色不同且加粗  
    for i, losses in enumerate(data):  
        ax.plot(epochs, losses, label=labels[i], linewidth=2.5)  
      
    # 添加网格线  
    ax.grid(True)  
      
    # 添加标题和标签  
    ax.set_title(title)  
    ax.set_xlabel(xlabel)  
    ax.set_ylabel(ylabel)  
      
    # 添加图例  
    ax.legend()  
      
    # 显示图形  
    plt.show()  

def plot_multiple_comparisons(predictions_list, labels_list, stocks):  
    """  
    绘制多组预测值和真实值的比较图。  
  
    参数:  
    predictions_list (list of lists): 包含多组预测值的列表。  
    labels_list (list of lists): 包含多组真实值的列表。  
  
    返回:  
    无  
    """  
    # 检查输入是否有效  
    if not predictions_list or not labels_list or len(predictions_list) != len(labels_list):  
        raise ValueError("predictions_list 和 labels_list 必须是长度相同的非空列表")  
      
    # 假设所有组的长度相同，我们可以获取任意一组的长度来设置x轴  
    x = range(len(predictions_list[0]))  
  
    # 创建图形和子图，nrows表示子图的行数  
    fig, axes = plt.subplots(nrows=len(predictions_list), figsize=(10, len(predictions_list) * 4))  
  
    # 如果只有一组数据，axes将是一个轴对象，而不是轴对象数组，需要处理这种情况  
    if len(predictions_list) == 1:  
        axes = [axes]  
  
    # 遍历每组数据并绘制子图  
    for i, (predictions, labels) in enumerate(zip(predictions_list, labels_list)):  
        ax = axes[i]  
        ax.plot(x, predictions, label='预测值', marker='o', linestyle='-', color='b')  
        ax.plot(x, labels, label='真是值', marker='^', linestyle='-', color='r')  
        ax.legend()  
        ax.set_title(f'{stocks[i]}')  
        # ax.set_xlabel('Time Index or Timestamp')  
        ax.set_ylabel('收盘价格')  
        ax.grid(True)  
  
    # 调整布局以防止子图重叠（可选）  
    plt.tight_layout()  
  
    # 显示图形  
    plt.show()  

def calculate_returns(predicted_prices, actual_prices, threshold=0.005):
    total_returns_predicted = []
    total_returns_actual = []
    for i in range(len(predicted_prices)):
        current_investment_predicted = 1  # 初始投资100%
        current_investment_actual = 1     # 初始投资100%
        for j in range(len(predicted_prices[i]) - 1):
            # 计算预测收益率
            predicted_return = (predicted_prices[i][j + 1] - predicted_prices[i][j]) / predicted_prices[i][j]
            # 计算实际收益率
            actual_return = (actual_prices[i][j + 1] - actual_prices[i][j]) / actual_prices[i][j]
            # 判断是否满足买入条件
            if predicted_return > threshold:
                current_investment_predicted *= (1 + predicted_return)  # 买入并计算收益
            else:
                current_investment_predicted *= (1 + actual_return)  # 不买入，根据实际收益计算
            current_investment_actual *= (1 + actual_return)  # 实际收益
        total_returns_predicted.append(current_investment_predicted - 1)  # 总收益率（预测）
        total_returns_actual.append(current_investment_actual - 1)        # 总收益率（实际）
    return total_returns_predicted, total_returns_actual

if __name__ == "__main__":

    from datasets import  get_DataLoader
      
    #  参数初始化
    opt = OptInit()
    opt.initialize()
    # USER_TOKEN = 'ad470017ef0a1f4d9613fb95712158e61352b39a258ec63cdf9a17cd'

    stocks = ['600519_SH.csv', '601398_SH.csv', '601288_SH.csv', '601857_SH.csv', '601988_SH.csv']  
    stock_loss, stock_prediction, stock_label = [], [], []
    for stock in stocks:
        DATA_PATH = f"data\\{stock}"
        close_max, close_min, train_dataloader, test_dataloader = get_DataLoader(DATA_PATH, opt.args)
        train_loss, prediction, label = main(train_dataloader, test_dataloader, opt.args, close_max, close_min)
        stock_loss.append(train_loss)
        stock_prediction.append(prediction)
        stock_label.append(label)

    predicted_returns, actual_returns = calculate_returns(stock_prediction, stock_label)
    # 输出结果
    for idx in range(len(stock_prediction)):
        print(f"股票{idx + 1}的总预测收益率: {predicted_returns[idx]:.2%}")
        print(f"股票{idx + 1}的总实际收益率: {actual_returns[idx]:.2%}")
    # 绘制loss曲线  
    plot_loss_curves(stock_loss, labels=stocks)
    # 绘制股票预测值与真实值对比图 
    plot_multiple_comparisons(stock_prediction, stock_label, stocks)

    print("finish")