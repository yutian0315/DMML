import os

DEFAULT_PICTURE_SIZE = (256, 256) #设置默认图像大小为256*256像素
DATASET_DIR = "../pythonProject/PlantVillage/"

# 模型参数
EPOCH = 10 #训练轮数设置
INIT_LR = 1e-4 #学习率
BATCH_SIZE = 32 #每批训练数据的大小
WIDTH, HEIGHT, DEPTH = 256, 256, 3 #输入图片的宽度,高度，深度
INPUT_SHAPE = (HEIGHT, WIDTH, DEPTH) #输入图片的尺寸
CHANDIM = -1 #通道维度

# 文件路径
LABEL_TRANSFORM_PATH = "label_transform.pkl" #标签编码器位置
BEST_MODEL_PATH = "bestModel.h5"#最好的模型保存位置及名称
PLOT_PATH = 'result.png' #准确率变化图片位置及名称
