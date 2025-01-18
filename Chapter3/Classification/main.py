import torch  
import torch.nn as nn  
import torch.optim as optim  
from torchvision import datasets, transforms, models  
from torch.utils.data import DataLoader, random_split  
import torch.nn.functional as F  
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft YaHei'  
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12

# 设备设置  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
# 数据预处理  
transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])  
  
# 加载数据集  
def load_data(batch_size=64):  
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)  
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)  
  
    # 划分训练集和验证集  
    num_train = len(trainset)  
    indices = list(range(num_train))  
    split = int(num_train * 0.8)  # 假设80%用于训练，20%用于验证  
    train_idx, val_idx = random_split(indices, [split, num_train - split])  
  
    trainset, valset = torch.utils.data.Subset(trainset, train_idx), torch.utils.data.Subset(trainset, val_idx)  
  
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)  
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)  
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)  
  
    return trainloader, valloader, testloader  
  
# 加载和修改ResNet18模型  
def get_model():  
    resnet18 = models.resnet18(weights=None)  
    num_ftrs = resnet18.fc.in_features  
    resnet18.fc = nn.Linear(num_ftrs, 10)  # 修改分类层为10类  
    return resnet18.to(device)  
  
# 训练模型  
def train_model(model, criterion, optimizer, trainloader, valloader, num_epochs=25):  
    best_acc = 0.0  
    best_model_wts = model.state_dict()  
    trainaccs, tranlosses = [], []
    valaccs, vallosses = [], []
    for epoch in range(num_epochs):  
        print(f'Epoch {epoch+1}/{num_epochs}')  
        print('-' * 10)  
  
        # 训练阶段  
        model.train() 
        correct = 0  
        total = 0
        train_loss = 0.0
        for inputs, labels in trainloader:  
            inputs, labels = inputs.to(device), labels.to(device)  
  
            optimizer.zero_grad()  
            outputs = model(inputs)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            # 计算准确率
            _, preds = torch.max(outputs, 1)  
            total += labels.size(0)  
            correct += (preds == labels).sum().item()  
            train_loss += loss.item() 
        train_acc = correct / total
        epoch_loss = train_loss / len(trainloader)  
        print(f'Train Loss: {epoch_loss:.4f}')  
        trainaccs.append(train_acc)
        tranlosses.append(epoch_loss)
        # 验证阶段  
        model.eval()  
        correct = 0  
        total = 0
        val_loss = 0.0
        with torch.no_grad():  
            for inputs, labels in valloader:  
                inputs, labels = inputs.to(device), labels.to(device)  
                outputs = model(inputs)
                loss = criterion(outputs, labels)    
                _, preds = torch.max(outputs, 1)  
                total += labels.size(0)  
                correct += (preds == labels).sum().item()  
                val_loss += loss.item()
        val_acc = correct / total  
        print(f'Val Acc: {val_acc:.4f}')  
        valaccs.append(val_acc)
        vallosses.append(val_loss / len(valloader))
        # 保存最佳模型  
        if val_acc > best_acc:  
            best_acc = val_acc  
            best_model_wts = model.state_dict()  
            torch.save(best_model_wts, 'best_model.pth')
    return trainaccs, tranlosses, valaccs, vallosses
# 测试模型  
def test_model(model, testloader):  
    model.eval()  
    correct = 0  
    total = 0  
    with torch.no_grad():  
        for inputs, labels in testloader:  
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)  
            _, preds = torch.max(outputs, 1)  
            total += labels.size(0)  
            correct += (preds == labels).sum().item()  
  
    print(f'Test Accuracy: {100 * correct / total:.2f}%')  
  
# 主程序  
def main():  
    trainloader, valloader, testloader = load_data(batch_size=64)  
    model = get_model()  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  
  
    # 训练模型并找到验证集上的最佳模型  
    train_accuracies, train_losses, val_accuracies, val_losses  = train_model(model, criterion, optimizer, trainloader, valloader, num_epochs=10)  
  

    # 绘制损失和准确率的折线图
    # 设置中文字体
    plt.rcParams['font.family'] = 'Microsoft YaHei'  
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者使用其他中文字体
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    epochs = range(0, 10)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='训练集', linewidth=2)
    plt.plot(epochs, val_losses, label='验证集', linewidth=2)
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.title('训练集和验证集损失')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='训练集', linewidth=2)
    plt.plot(epochs, val_accuracies, label='验证集', linewidth=2)
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.title('训练集和验证集准确率')

    plt.tight_layout()
    plt.show()
    plt.savefig("result.png")

    # 加载最佳模型状态  
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))  # 假设我们在某个地方保存了最佳模型为'best_model.pth'  
    
    # 测试最佳模型  
    test_model(model, testloader)  
  
if __name__ == '__main__':  
    main()  
  

