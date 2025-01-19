import matplotlib.pyplot as plt
from config import PLOT_PATH

def plotTrainingHistory(history):
    #获得history中的acc和valAcc，loss和valLoss数据
    acc = history.history['accuracy']
    valAcc = history.history['val_accuracy']
    loss = history.history['loss']
    valLoss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    #设置图片格式
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 12

    plt.figure(figsize=(12, 5))

    #绘制准确率图片
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='训练集准确率')
    plt.plot(epochs, valAcc, 'r', label='验证集准确率')
    plt.title('训练集和验证集的准确率变化')
    plt.legend()

    #绘制损失值图片
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='训练集损失')
    plt.plot(epochs, valLoss, 'r', label='验证集损失')
    plt.title('训练集和验证集的损失变化')
    plt.legend()
    plt.savefig(PLOT_PATH)
