from keras.models import Sequential
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense
from keras.optimizers import Adam
from config import INPUT_SHAPE, CHANDIM, INIT_LR, EPOCH


def createModel(classes):
    # CNN模型
    model = Sequential()
    model.add(Conv2D(8, (3, 3), padding="same", input_shape=INPUT_SHAPE))  # Conv2D层
    model.add(Activation("relu"))  # relu激活函数
    model.add(BatchNormalization(axis=CHANDIM))  # 批标准化（Batch Normalization）用于加速训练并提高模型的稳定性。它通过规范化每一层的输入，减少梯度消失和爆炸的风险。
    model.add(MaxPooling2D(pool_size=(3, 3)))  # 最大池化层（Max Pooling）用于减少空间维度（宽度和高度），从而减少计算量，并提取最重要的特征。pool_size=(3, 3) 表示池化窗口大小为 3x3。
    model.add(Dropout(0.25))  # 在每次训练时随机丢弃 25% 的神经元
    model.add(Flatten())  # 将多维的输入展平为一维向量
    model.add(Dense(524))  # 全连接层
    model.add(Activation("relu"))  # relu激活函数
    model.add(BatchNormalization())  # 再次使用批标准化层，默认为对最后一个轴（即通道维度）进行标准化。
    model.add(Dropout(0.5))  # 在每次训练时随机丢弃 50% 的神经元
    model.add(Dense(classes))  # 输出层
    model.add(Activation("softmax"))  # Softmax 激活函数用于多分类问题。它将输出值转换为概率分布，表示每个类别的概率。
    model.summary()  # 型的架构概览
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCH)#Adam 优化器
    # 定义一个回调函数，在验证准确率达到新高时保存模型
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model
