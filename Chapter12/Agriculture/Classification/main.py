from datasets import loadData
from models import createModel
from utils import plotTrainingHistory
from config import BEST_MODEL_PATH, EPOCH, BATCH_SIZE
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

if __name__ == "__main__":
    xTrain, xVal, xTest, yTrain, yVal, yTest, numClasses = loadData()    # 加载数据
    model = createModel(numClasses)    # 初始化和编译模型
    # 设置检查点
    checkpoint = ModelCheckpoint(BEST_MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # 训练模型
    history = model.fit(
        xTrain,
        yTrain,
        batch_size= BATCH_SIZE,
        validation_data=(xVal, yVal),
        epochs=EPOCH,
        verbose=1,
        callbacks=[checkpoint]
    )

    plotTrainingHistory(history)    # 绘制训练过程
    model = load_model(BEST_MODEL_PATH)    # 加载最佳模型并评估
    print("训练集评估：", model.evaluate(xTrain, yTrain))
    print("测试集评估：", model.evaluate(xTest, yTest))
    print("验证集评估：", model.evaluate(xVal, yVal))
